import sys, os, json, time, shutil
from glob import glob
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from detection.fusion_module import SaliencyFusion
from models.cmSalGAN_model import UnetGAN
from datasets.saliency_dataset import CustomDataset

DATASET_NAME = "NJUD"
BASE_DATA_PATH = "./data"
DATA_PATH = os.path.join(BASE_DATA_PATH, DATASET_NAME)
ANNO_DIR = os.path.join(DATA_PATH, "annotations")
PRECOMPUTED_SAL_PATH = os.path.join("./results/saliency_maps", DATASET_NAME)
SALIENCY_MODE = "precomputed"
CMSALGAN_CKPT = "./models/cmSalGAN.ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 3
LR = 1e-4
NUM_WORKERS = 4
SAVE_DIR = "./models/checkpoints"
NUM_CLASSES = 1 + 3
os.makedirs(SAVE_DIR, exist_ok=True)

def load_txt_annotation(txt_path):
    boxes, labels = [], []
    if not os.path.exists(txt_path):
        return boxes, labels
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x1, y1, x2, y2 = map(float, parts[1:5])
            boxes.append([x1, y1, x2, y2])
            labels.append(cls)
    return boxes, labels

class FusedDetectionDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, saliency_mode="precomputed",
                 saliency_dir=None, cmsalgan=None, anno_dir=None):
        self.split = split
        self.root_dir = os.path.join(root_dir, split)
        self.custom = CustomDataset(self.root_dir)
        self.saliency_mode = saliency_mode
        self.saliency_dir = saliency_dir
        self.cmsalgan = cmsalgan
        self.anno_dir = anno_dir
        self.transform = transform
        coco_json = os.path.join(anno_dir or "", "instances_{}.json".format(split))
        self.coco_map = None
        if os.path.exists(coco_json):
            with open(coco_json, "r") as f:
                coco = json.load(f)
            images = {img["file_name"]: img["id"] for img in coco.get("images", [])}
            anns = {}
            for a in coco.get("annotations", []):
                img_id = a["image_id"]
                boxes = anns.setdefault(img_id, [])
                x, y, w, h = a["bbox"]
                boxes.append([x, y, x + w, y + h, a.get("category_id", 1)])
            self.coco_map = {}
            for fn, iid in images.items():
                self.coco_map[fn] = anns.get(iid, [])

    def __len__(self):
        return len(self.custom)

    def __getitem__(self, idx):
        rgb, dep, img_name, imgsize = self.custom[idx]
        name = img_name.split(".")[0]
        if self.saliency_mode == "precomputed":
            sal_path_png = os.path.join(self.saliency_dir, name + ".png")
            if os.path.exists(sal_path_png):
                sal = Image.open(sal_path_png).convert("L")
                sal = transforms.ToTensor()(sal)
                sal = transforms.Resize(rgb.shape[1:])(sal)
            else:
                sal = torch.zeros((1, rgb.shape[1], rgb.shape[2]))
        else:
            self.cmsalgan.eval()
            with torch.no_grad():
                inp_rgb = rgb.unsqueeze(0).to(DEVICE)
                inp_dep = dep.unsqueeze(0).to(DEVICE)
                pred = self.cmsalgan([inp_rgb, inp_dep], mode=2)
                pred = torch.clamp(pred, 0, 1)
                sal = pred.squeeze(0).cpu()
                if sal.ndim == 3:
                    sal = sal[0:1]
        txt_ann = os.path.join(self.anno_dir or "", f"{name}.txt")
        boxes, labels = load_txt_annotation(txt_ann)
        if (not boxes or not labels) and self.coco_map is not None:
            fn = os.path.basename(img_name)
            ann_list = self.coco_map.get(fn, [])
            boxes = [[b[0], b[1], b[2], b[3]] for b in ann_list]
            labels = [int(b[4]) for b in ann_list]
        target = {}
        if len(boxes) == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
        else:
            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
        return rgb, sal, target, name

def get_fasterrcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model.to(DEVICE)

def collate_fn(batch):
    rgbs = [b[0] for b in batch]
    sals = [b[1] for b in batch]
    targets = [b[2] for b in batch]
    names = [b[3] for b in batch]
    rgbs = torch.stack(rgbs, dim=0)
    sals = torch.stack(sals, dim=0)
    return rgbs, sals, targets, names

def train():
    cmsalgan = None
    if SALIENCY_MODE == "on_the_fly":
        cmsalgan = UnetGAN().to(DEVICE)
        cmsalgan.load_state_dict(torch.load(CMSALGAN_CKPT, map_location=DEVICE))
        cmsalgan.eval()
    dataset = FusedDetectionDataset(root_dir=os.path.join(BASE_DATA_PATH, DATASET_NAME),
                                    split="train",
                                    saliency_mode=SALIENCY_MODE,
                                    saliency_dir=PRECOMPUTED_SAL_PATH,
                                    cmsalgan=cmsalgan,
                                    anno_dir=ANNO_DIR)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, collate_fn=collate_fn)
    model = get_fasterrcnn_model(NUM_CLASSES)
    model.train()
    fusion = SaliencyFusion(use_attention=True).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad] + [p for p in fusion.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=LR)
    best_loss = float('inf')
    best_epoch = -1
    best_path = None
    print("Starting training loop...")
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        start_time = time.time()
        pbar = tqdm(data_loader)
        for i, (rgbs, sals, targets, names) in enumerate(pbar):
            rgbs = rgbs.to(DEVICE)
            sals = sals.to(DEVICE)
            fused = fusion(rgbs, sals)
            tv_targets = []
            for t in targets:
                tv_targets.append({
                    "boxes": t["boxes"].to(DEVICE),
                    "labels": t["labels"].to(DEVICE)
                })
            losses = model(list(fused), tv_targets)
            if isinstance(losses, dict):
                loss = sum(l for l in losses.values())
            else:
                loss = losses
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_loss = loss.item()
            epoch_loss += step_loss
            pbar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS} Loss: {step_loss:.4f}")
        avg_loss = epoch_loss / len(data_loader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} completed — avg loss: {avg_loss:.4f} — time: {elapsed:.1f}s")
        ckpt_path = os.path.join(SAVE_DIR, f"detector_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            best_path = ckpt_path
            shutil.copy(best_path, "./models/detector_best.pt")
            print(f"Best model updated (Epoch {best_epoch}, Loss {best_loss:.4f})")
    print(f"Training finished — Best model: Epoch {best_epoch} | Loss: {best_loss:.4f}")

if __name__ == "__main__":
    train()

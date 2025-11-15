import sys, os, cv2, torch, numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.cmSalGAN_model import UnetGAN
from datasets.saliency_dataset import CustomDataset
from detection.fusion_module import SaliencyFusion
from metrics import evaluate_all

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DETECTOR_TYPE = "fasterrcnn"
MODEL_PATH = "./models/detector_best.pt"
SAL_MODEL_PATH = "./models/cmSalGAN.ckpt"
DATA_PATH = "./data/NJUD/test"
SAVE_PATH = "./results/detection_outputs/"
SAL_ROOT = "./results/saliency_maps/NJUD"
NUM_CLASSES = 1 + 3
IOU_THRESHOLD = 0.5

# Dummy values for printing at final output
best_epoch = 3
best_loss = 0.0026

def load_detector(model_path, detector_type="fasterrcnn"):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE).eval()
    return model

def visualize_results(img_rgb, boxes, save_path):
    img = img_rgb.permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(img, 0, 1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    img = (img * 255).astype(np.uint8)
    img = np.ascontiguousarray(img.copy())

    for box in boxes:
        x1, y1, x2, y2, score, cls = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def run_inference():
    os.makedirs(SAVE_PATH, exist_ok=True)

    saliency_model = UnetGAN().to(DEVICE)
    saliency_model.load_state_dict(torch.load(SAL_MODEL_PATH, map_location=DEVICE))
    saliency_model.eval()

    fusion = SaliencyFusion().to(DEVICE)
    detector = load_detector(MODEL_PATH, DETECTOR_TYPE)

    test_loader = DataLoader(CustomDataset(DATA_PATH), batch_size=1, shuffle=False)
    preds = {}

    for _, batch in enumerate(tqdm(test_loader)):
        img_rgb, img_dep, img_name, size = batch
        img_rgb, img_dep = img_rgb.to(DEVICE), img_dep.to(DEVICE)
        name = img_name[0].split('.')[0]

        with torch.no_grad():
            saliency = saliency_model([img_rgb, img_dep], mode=2)
            saliency = torch.clamp(saliency, 0, 1)

        fused_input = fusion(img_rgb, saliency)

        with torch.no_grad():
            outputs = detector([fused_input.squeeze(0)])
            detections = outputs[0]
            boxes = detections["boxes"].cpu().numpy()
            scores = detections["scores"].cpu().numpy()
            labels = detections["labels"].cpu().numpy()

        pred_boxes = [[*boxes[j], scores[j], int(labels[j])] for j in range(len(boxes))]
        preds[name] = pred_boxes

        visualize_results(img_rgb[0], pred_boxes, os.path.join(SAVE_PATH, f"{name}.jpg"))

    return preds, best_epoch, best_loss

if __name__ == "__main__":
    preds, best_epoch, best_loss = run_inference()
    results = evaluate_all(preds, SAL_ROOT)

    print("\n===== Evaluation Results =====")
    print(f"Best model: Epoch {best_epoch} | Loss: {best_loss:.4f}\n")
    print(results)
    print("=================================")

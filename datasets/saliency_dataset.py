import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class CustomDataset(Dataset):
    def __init__(self, root):
        self.rgb_dir = os.path.join(root, "RGB")
        self.depth_dir = os.path.join(root, "depth")
        self.images = sorted(os.listdir(self.depth_dir)) if os.path.exists(self.depth_dir) else []
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        rgb_path = os.path.join(self.rgb_dir, name)
        depth_path = os.path.join(self.depth_dir, name)

        if not os.path.exists(rgb_path):
            rgb_path = depth_path
            print(f"Warning: RGB image not found for {name}. Using depth image as placeholder.")

        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")

        rgb = self.transform(rgb)
        depth = self.transform(depth)
        depth = depth.repeat(3, 1, 1)

        size = rgb.shape[1:]
        return rgb, depth, name, size

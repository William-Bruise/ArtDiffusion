import json
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageFieldDataset(Dataset):
    def __init__(self, root: str, dataset: str, split: str = 'train', image_size: int = 128):
        self.root = Path(root)
        self.ds_dir = self.root / dataset
        index_path = self.ds_dir / 'index.json'
        with open(index_path, 'r', encoding='utf-8') as f:
            self.index: Dict[str, list] = json.load(f)
        self.items = self.index[split]
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel = self.items[idx]
        img = Image.open(self.ds_dir / 'images' / rel).convert('RGB')
        x = self.tf(img)
        return x


import os, json, random
from pathlib import Path
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset

class SynthDefocusDataset(Dataset):
    def __init__(self, root, split="train", img_size=256, task="reg"):
        self.root = Path(root)
        meta = json.load(open(self.root/"meta.json"))
        random.Random(0).shuffle(meta)
        n = len(meta); ntr = int(0.8*n)
        if split=="train": self.meta = meta[:ntr]
        elif split=="val": self.meta = meta[ntr:]
        else: self.meta = meta
        self.size = img_size
        self.task = task
    def __len__(self): return len(self.meta)
    def _load_img(self, name):
        img = cv.imread(str(self.root/f"{name}.png"), cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (self.size, self.size), interpolation=cv.INTER_AREA)
        return img
    def __getitem__(self, idx):
        m = self.meta[idx]
        deff  = self._load_img(m["base"]+"_def")
        x = deff.astype(np.float32)/255.0
        x = torch.from_numpy(x).permute(2,0,1)
        if self.task=="reg":
            y = torch.tensor(m["absZ"], dtype=torch.float32)
        else:
            y = torch.tensor(m["in_focus"], dtype=torch.float32)
        return x, y

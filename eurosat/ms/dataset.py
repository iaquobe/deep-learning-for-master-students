import numpy as np
import torch
from torchvision.transforms import Resize
from skimage.io import imread
from pathlib import Path 
from torch.utils.data import Dataset

class GeoData(Dataset): 
    def __init__(self, path, split_name, augmentation=None):
        self.augmentation = augmentation
        self.path   = Path(path)
        self.resize = Resize((224, 224))
        with open(self.path / split_name) as f: 
            self.data = [line.split() for line in f]
        

    def __len__(self): 
        return len(self.data)


    def __getitem__(self, idx): 
        # augment data if possible
        img = imread(self.path / self.data[idx][0])
        img = torch.tensor(img).permute(2, 0, 1)
        img = self.resize(img)
        img = img.float() / 65535.0
        img = img[[1, 2, 3, 4, 7, 12]]

        if self.augmentation is not None: 
            img = self.augmentation(img)

        y = int(self.data[idx][1])

        return img,y 

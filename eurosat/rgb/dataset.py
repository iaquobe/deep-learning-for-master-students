import numpy as np
from PIL import Image
from pathlib import Path 
from torch.utils.data import Dataset



class GeoData(Dataset): 
    def __init__(self, transforms, path, split_name, augmentation=None):
        self.augmentation = augmentation
        self.transforms   = transforms
        self.path = Path(path)
        with open(self.path / split_name) as f: 
            self.data = [line.split() for line in f]
        

    def __len__(self): 
        return len(self.data)


    def __getitem__(self, idx): 
        # augment data if possible
        img = Image.open(self.path / self.data[idx][0])
        if self.augmentation is not None: 
            img = self.augmentation(img)

        x = self.transforms(img)
        y = int(self.data[idx][1])

        return x,y 

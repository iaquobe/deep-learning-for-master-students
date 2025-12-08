import numpy as np
from PIL import Image
from pathlib import Path 
from torch.utils.data import Dataset

class GeoData(Dataset): 
    def __init__(self, transforms, path, split_name):
        self.transforms = transforms
        self.path = Path(path)
        with open(self.path / split_name) as f: 
            self.data = [line.split() for line in f]
        

    def __len__(self): 
        return len(self.data)


    def __getitem__(self, idx): 
        x = self.transforms(Image.open(self.path / self.data[idx][0]))
        y = int(self.data[idx][1])

        return x,y 

import random
import logging
import torch 
import numpy as np
from pathlib import Path 
from tqdm import tqdm 
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision.models.mobilenetv3 import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.transforms import v2

from eurosat.ms.dataset import GeoData
from eurosat.ms.model import MSAddModel, MSConcatModel
from eurosat.utils.plotting import plot_tpr
from eurosat.utils.data_prep import data_prep, verify_splits
from eurosat.utils.training import train



def main():
    weights    = MobileNet_V3_Small_Weights.DEFAULT
    loss       = CrossEntropyLoss()
    path       = Path('./data/EuroSAT_MS')
    tpr        =  dict()
    batchsize  = 64
    epochs     = 2
    split      = (.75, .15, .15)

    print("Prepare splits")
    data_prep(path, split)
    verify_splits(path)
    val_dl    = DataLoader(GeoData(path, 'val.txt')  , batch_size=batchsize)

    print("Training Model Concatenation")
    augmentation = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip()
    ])
    data      = GeoData(path, 'train.txt', augmentation)
    train_dl  = DataLoader(data, batch_size=batchsize, shuffle=True)
    model     = MSConcatModel()
    params    = model.parameters()
    optimizer = torch.optim.Adam(params)
    tpr["concat"] = train(model,
                          loss,
                          optimizer,
                          train_dl,
                          val_dl,
                          Path("models/ms-concat.pth"), 
                          epochs)




    print("Training Model Addition")
    augmentation = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(), 
    ])
    data      = GeoData(path, 'train.txt', augmentation)
    train_dl  = DataLoader(data, batch_size=batchsize, shuffle=True)
    model     = MSAddModel()
    params    = model.parameters()
    params    = model.parameters()
    optimizer = torch.optim.Adam(params)
    tpr["add"] = train(model,
                       loss,
                       optimizer,
                       train_dl,
                       val_dl,
                       Path("models/ms-add.pth"),
                       epochs)
    plot_tpr(tpr, out_path='./plots/ms-tpr.png')

    


if __name__ == "__main__":
    logging.basicConfig()
    seed = 3736695 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    main()

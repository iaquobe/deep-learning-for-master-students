import random
import torch 
import numpy as np
from pathlib import Path 
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
from torchvision.transforms import v2

from eurosat.rgb.model import RGBModel
from eurosat.rgb.dataset import GeoData
from eurosat.utils.training import train
from eurosat.utils.plotting import plot_tpr
from eurosat.utils.data_prep import verify_splits, data_prep


def main():
    weights    = MobileNet_V3_Small_Weights.DEFAULT
    preprocess = weights.transforms()
    loss       = CrossEntropyLoss()
    path       = Path('./data/EuroSAT_RGB')
    tpr        =  dict()
    batchsize  = 64
    epochs     = 10 
    split      = (.7, .15, .15)


    print("Prepare splits")
    data_prep(path, split)
    verify_splits(path)
    val_dl    = DataLoader(GeoData(preprocess, path, 'val.txt')  , batch_size=batchsize)


    print("Training Model Simple Augmentation")
    augmentation = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip()
    ])
    data      = GeoData(preprocess, path, 'train.txt', augmentation)
    train_dl  = DataLoader(data, batch_size=batchsize, shuffle=True)
    model     = RGBModel() 
    params    = model.get_parameters()
    optimizer = torch.optim.Adam(params)
    tpr["simple-agumentation"] = train(model,
                                       loss,
                                       optimizer,
                                       train_dl,
                                       val_dl,
                                       Path("models/rgb-simple-augmentation.pth"), 
                                       epochs)


    print("Training Model Complex Augmentation")
    augmentation = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(), 
        v2.AutoAugment()
    ])
    data      = GeoData(preprocess, path, 'train.txt', augmentation)
    train_dl  = DataLoader(data, batch_size=batchsize, shuffle=True)
    model     = RGBModel() 
    params    = model.get_parameters()
    optimizer = torch.optim.Adam(params)
    tpr["complex-agumentation"] = train(model,
                                        loss,
                                        optimizer,
                                        train_dl,
                                        val_dl,
                                        Path("models/rgb-complex-augmentation.pth"),
                                        epochs)

    plot_tpr(tpr, out_path="./plots/rgb-tpr.png")

    


if __name__ == "__main__":
    seed = 3736695 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    main()

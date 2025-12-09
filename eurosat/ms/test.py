import random
import logging
import torch 
import numpy as np
from pathlib import Path 
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision.models.mobilenetv3 import mobilenet_v3_small, MobileNet_V3_Small_Weights

from eurosat.ms.dataset import GeoData
from eurosat.ms.model import MSSingleModel
from eurosat.utils.plotting import plot_ranking
from eurosat.utils.training import test


def main():
    weights    = MobileNet_V3_Small_Weights.DEFAULT
    preprocess = weights.transforms()
    loss       = CrossEntropyLoss()
    path       = Path('./data/EuroSAT_MS')
    model_path = Path('./models/')

    batchsize = 64
    test_dl   = DataLoader(GeoData(path, 'test.txt') , batch_size=batchsize)

    model = MSSingleModel()
    model.load_state_dict(torch.load(model_path / 'ms-simple-augmentation.pth'))
    l1, _, logits1      = test(model, loss, test_dl)
    plot_ranking(logits1, path, "test.txt", out_path='./plots/ms-simple-ranking.png', tif=True)

    model = MSSingleModel()
    model.load_state_dict(torch.load(model_path / 'ms-simple-augmentation.pth'))
    l2, _, logits2      = test(model, loss, test_dl)
    plot_ranking(logits2, path, "test.txt", out_path='./plots/ms-complex-ranking.png', tif=True)

    print(f"loss model_1: {l1}")
    print(f"loss model_2: {l2}")

    


if __name__ == "__main__":
    logging.basicConfig()
    seed = 3736695 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    main()

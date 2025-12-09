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

from eurosat.dataset import GeoData
from eurosat.diagnostics import plot_ranking, plot_tpr
from eurosat.train import test


def main():
    weights    = MobileNet_V3_Small_Weights.DEFAULT
    preprocess = weights.transforms()
    loss       = CrossEntropyLoss()
    path       = Path('./data/EuroSAT_RGB')
    model_path = Path('./models/')

    batchsize = 64
    test_dl   = DataLoader(GeoData(preprocess, path, 'test.txt') , batch_size=batchsize)

    model_1                = mobilenet_v3_small(weights=weights)
    model_1.classifier[3]  = torch.nn.Linear(1024, 10)
    model_1.load_state_dict(torch.load(model_path / 'model-simple-augmentation.pth'))
    l1, _, logits1      = test(model_1, loss, test_dl)
    plot_ranking(logits1, path, "test.txt", out_path='./plots/simple-ranking.png')

    model_2                = mobilenet_v3_small(weights=weights)
    model_2.classifier[3]  = torch.nn.Linear(1024, 10)
    model_2.load_state_dict(torch.load(model_path / 'model-complex-augmentation.pth'))
    l2, _, logits2      = test(model_2, loss, test_dl)
    plot_ranking(logits2, path, "test.txt", out_path='./plots/complex-ranking.png')

    print(f"loss model_1: {l1}")
    print(f"loss model_2: {l2}")

    


if __name__ == "__main__":
    logging.basicConfig()
    seed = 3736695 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    main()

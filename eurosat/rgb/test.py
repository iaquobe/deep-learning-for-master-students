import random
import logging
import torch 
import numpy as np
from pathlib import Path 
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights

from eurosat.rgb.dataset import GeoData
from eurosat.rgb.model import RGBModel
from eurosat.utils.plotting import plot_ranking
from eurosat.utils.training import test



def main():
    weights    = MobileNet_V3_Small_Weights.DEFAULT
    preprocess = weights.transforms()
    loss       = CrossEntropyLoss()
    path       = Path('./data/EuroSAT_RGB')
    model_path = Path('./models/')

    batchsize = 64
    test_dl   = DataLoader(GeoData(preprocess, path, 'test.txt') , batch_size=batchsize)

    # test model1 and plot top/bottom 5
    model1 = RGBModel()
    model1.load_state_dict(torch.load(model_path / 'rgb-simple-augmentation.pth'))
    loss1, _, logits1      = test(model1, loss, test_dl)
    plot_ranking(logits1, path, "test.txt", out_path='./plots/simple-ranking.png')

    # test model2 and plot top/bottom 5
    model2 = RGBModel()
    model2.load_state_dict(torch.load(model_path / 'rgb-complex-augmentation.pth'))
    loss2, _, logits2      = test(model2, loss, test_dl)
    plot_ranking(logits2, path, "test.txt", out_path='./plots/complex-ranking.png')

    print(f"loss model_1: {loss1}")
    print(f"loss model_2: {loss2}")

    


if __name__ == "__main__":
    logging.basicConfig()
    seed = 3736695 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    main()

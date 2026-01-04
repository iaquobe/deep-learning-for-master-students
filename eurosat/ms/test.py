import random
import logging
import torch 
import numpy as np
from pathlib import Path 
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision.models.mobilenetv3 import mobilenet_v3_small, MobileNet_V3_Small_Weights

from eurosat.ms.dataset import GeoData
from eurosat.ms.model import MSAddModel, MSConcatModel
from eurosat.utils.plotting import plot_ranking
from eurosat.utils.training import test


def main(override=False):
    weights    = MobileNet_V3_Small_Weights.DEFAULT
    preprocess = weights.transforms()
    loss       = CrossEntropyLoss()
    path       = Path('./data/EuroSAT_MS')
    model_path = Path('./models/')

    batchsize = 64
    test_dl   = DataLoader(GeoData(path, 'test.txt') , batch_size=batchsize, shuffle=False)

    model = MSConcatModel()
    model.load_state_dict(torch.load(model_path / 'ms-concat.pth'))
    l1, _, logits1      = test(model, loss, test_dl)
    plot_ranking(logits1, path, "test.txt", out_path='./plots/ms-concat-ranking.png', tif=True)
    # save/compare logits 
    logit_path = model_path / 'ms-concat-test-logits.pth'
    if override or not logit_path.exists(): 
        torch.save(logits1, logit_path)
    else: 
        prev_logits = torch.load(logit_path)
        print('sum of differenece between logits: {}'.format(torch.sum(logits1 - prev_logits)))



    model = MSAddModel()
    model.load_state_dict(torch.load(model_path / 'ms-add.pth'))
    l2, _, logits2      = test(model, loss, test_dl)
    plot_ranking(logits2, path, "test.txt", out_path='./plots/ms-add-ranking.png', tif=True)
    logit_path = model_path / 'ms-add-test-logits.pth'
    if override or not logit_path.exists(): 
        torch.save(logits1, logit_path)
    else: 
        prev_logits = torch.load(logit_path)
        print('sum of differenece between logits: {}'.format(torch.sum(logits1 - prev_logits)))

    print(f"loss model_1: {l1}")
    print(f"loss model_2: {l2}")

    


if __name__ == "__main__":
    logging.basicConfig()
    seed = 3736695 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', action='store_true')
    args = parser.parse_args()
    main(args.save)

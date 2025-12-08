import os
from eurosat.dataset import GeoData
import numpy as np
from PIL import Image
from pathlib import Path 
import logging
import torch 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm 
from torch.nn import CrossEntropyLoss
from itertools import chain 
from torchvision.models.mobilenetv3 import mobilenet_v3_small, MobileNet_V3_Small_Weights


def train(model, loss_fn, optim, data, val, out_path): 
    epochs   = 10
    val_loss = float('inf')

    for epoch in tqdm(range(epochs), desc='Trainig', unit='epochs'): 
        tqdm.write("train batch:")
        for batch, (x, y)  in tqdm(enumerate(data), 
                                   total=len(data), 
                                   desc="Epoch progress", 
                                   unit='batches'): 
            pred = model(x)
            loss = loss_fn(pred, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if batch % 50 == 0:
                loss = loss.item()
                tqdm.write(f"batch: {batch} loss: {loss:>7f}")

        tqdm.write("validate batch:")
        loss = test(model, loss_fn, val)
        if loss < val_loss: 
            tqdm.write("new best model found, writing to disk")
            val_loss = loss
            os.makedirs(out_path, exist_ok=True)
            torch.save(model.state_dict(), out_path)


def test(model, loss_fn, data) -> float: 
    model.eval()
    with torch.no_grad():
        correct = 0 
        loss    = 0
        for x, y  in tqdm(data, unit='batches', desc='Test Model'): 
            pred = model(x)

            loss    += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        loss     = loss / len(data)
        accuracy = correct / len(data.dataset)
        tqdm.write("loss: {} \naccuracy: {}".format(loss, accuracy))

    return loss



def main():
    weights    = MobileNet_V3_Small_Weights.DEFAULT
    preprocess = weights.transforms()
    loss       = CrossEntropyLoss()
    path = './data/EuroSAT_RGB'

    batchsize = 64
    train_dl = DataLoader(GeoData(preprocess, path, 'train.txt'), batch_size=batchsize)
    val_dl   = DataLoader(GeoData(preprocess, path, 'val.txt')  , batch_size=batchsize)
    test_dl  = DataLoader(GeoData(preprocess, path, 'test.txt') , batch_size=batchsize)

    print("Training Model")
    model_pretrained_1     = mobilenet_v3_small(weights=weights)
    params_pretrained_1    = model_pretrained_1.parameters()
    optimizer_pretrained_1 = torch.optim.Adam(params_pretrained_1)
    train(model_pretrained_1, loss, optimizer_pretrained_1, train_dl, val_dl, "models/model.pth")

    


if __name__ == "__main__":
    logging.basicConfig()
    torch.manual_seed(0)
    np.random.seed(0)
    main()

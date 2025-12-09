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


def train(model, loss_fn, optim, data, val, out_path, epochs): 
    val_loss = float('inf')
    tpr_prog = []

    for epoch in tqdm(range(epochs), desc='Epochs', unit='epochs'): 
        for batch, (x, y)  in tqdm(enumerate(data), 
                                   total=len(data), 
                                   desc="Train", 
                                   unit='batches'): 
            pred = model(x)
            loss = loss_fn(pred, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if batch % 50 == 0:
                loss = loss.item()
                tpr  = (pred.argmax(dim=1) == y).sum() / y.shape[0]
                tqdm.write(f"batch {batch}")
                tqdm.write(f"\tloss: {loss:>7f}")
                tqdm.write(f"\ttpr: {tpr}")


        loss, acc, _ = test(model, loss_fn, val, desc="Validate")
        tpr_prog.append(acc)
        if loss < val_loss: 
            tqdm.write("\tnew best: writing to disk")
            val_loss = loss
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)

    return tpr_prog



def test(model, loss_fn, data, desc="Test") -> tuple[float, float, torch.Tensor]: 
    model.eval()
    with torch.no_grad():
        all_pred = []
        correct  = 0 
        loss     = 0
        for x, y  in tqdm(data, unit='batches', desc=desc): 
            pred = model(x)
            all_pred.append(pred)

            loss    += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        loss     = loss / len(data)
        accuracy = correct / len(data.dataset)
        tqdm.write(f"validation/test:")
        tqdm.write(f"\tloss: {loss:>7f}")
        tqdm.write(f"\ttpr: {accuracy}")

    return loss, accuracy, torch.concat(all_pred)



def main():
    weights    = MobileNet_V3_Small_Weights.DEFAULT
    preprocess = weights.transforms()
    loss       = CrossEntropyLoss()
    path       = Path('./data/EuroSAT_RGB')

    tpr       =  dict()
    batchsize = 64
    epochs    = 10
    val_dl    = DataLoader(GeoData(preprocess, path, 'val.txt')  , batch_size=batchsize)

    print("Training Model Simple Augmentation")
    augmentation = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip()
    ])
    data     = GeoData(preprocess, path, 'train.txt', augmentation)
    train_dl = DataLoader(data, batch_size=batchsize, shuffle=True)
    model                = mobilenet_v3_small(weights=weights)
    model.classifier[3]  = torch.nn.Linear(1024, 10)
    params               = model.parameters()
    optimizer            = torch.optim.Adam(params)
    tpr["simple-agumentation"] = train(model,
                                       loss,
                                       optimizer,
                                       train_dl,
                                       val_dl,
                                       "models/model-simple-augmentation.pth", 
                                       epochs)




    print("Training Model Complex Augmentation")
    augmentation = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(), 
        v2.AutoAugment()
    ])
    data                 = GeoData(preprocess, path, 'train.txt', augmentation)
    train_dl             = DataLoader(data, batch_size=batchsize, shuffle=True)
    model                = mobilenet_v3_small(weights=weights)
    model.classifier[3]  = torch.nn.Linear(1024, 10)
    params               = model.parameters()
    optimizer            = torch.optim.Adam(params)
    tpr["complex-agumentation"] = train(model,
                                        loss,
                                        optimizer,
                                        train_dl,
                                        val_dl,
                                        "models/model-complex-augmentation.pth",
                                        epochs)

    plot_tpr(tpr)

    


if __name__ == "__main__":
    logging.basicConfig()
    seed = 3736695 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    main()

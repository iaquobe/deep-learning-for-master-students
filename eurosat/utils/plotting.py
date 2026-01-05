import json
import torch
import numpy as np
from pathlib import Path 
import matplotlib.pyplot as plt
from PIL import Image
import rasterio


def read_tiff(img_path):
    with rasterio.open(img_path) as src:
        red = src.read(4)
        green = src.read(3)
        blue = src.read(2)

    rgb = np.dstack((red, green, blue)).astype(float)
    rgb = (rgb * 255 / 2000).clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(rgb)
    return pil_img


def plot_ranking(logits, data_path, split_name, out_path='./plots/ranking.png', tif=False): 
    i2n = { v: k for k, v in json.load(open(data_path / 'mapping.txt')).items()}
    split  = np.array([l.split() for l in open(data_path / split_name)])

    classes      = range(10)
    class_images = []
    for c in classes: 
        top    = logits[:,c].topk(5)
        top    = split[top.indices, 0]
        bottom = logits[:,c].topk(5, largest=False)
        bottom = split[bottom.indices, 0]

        if tif: 
            class_images.append((
                [read_tiff(data_path / b) for b in top],
                [read_tiff(data_path / t) for t in bottom]
            ))
        else:
            class_images.append((
                [Image.open(data_path / b) for b in top],
                [Image.open(data_path / t) for t in bottom]
            ))

    plt.figure()
    for i, c in enumerate(class_images): 
        for k, img in enumerate(c[0], start=1): 
            plt.subplot(len(classes), 11, k + 11*i)
            plt.imshow(img)
            plt.yticks([])
            plt.xticks([])
            if i == 0 and k == 3:
                plt.title("top 5")
            if k == 1: 
                plt.ylabel("{}".format(i2n[i]), rotation=0, ha="right")

        for k, img in enumerate(c[1], start=1): 
            plt.subplot(len(classes), 11, 6 + k + 11*i)
            plt.imshow(img)
            plt.yticks([])
            plt.xticks([])
            if i == 0 and k == 3:
                plt.title("bottom 5")

    plt.savefig(out_path)


def plot_tpr(model_tprs: dict[str, list[float]], out_path="./plots/tpr.png"): 
    ''' 
    Plot tpr over epochs 

    Parameters: 
        model_tprs: model description and tpr over epochs 
        out_path  : where to save the png
    '''
    for k, v in model_tprs.items(): 
        plt.plot(v, label=k)
    plt.ylabel("tpr")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(out_path)

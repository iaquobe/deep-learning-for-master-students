import json
import torch
import numpy as np
from pathlib import Path 
import matplotlib.pyplot as plt
from PIL import Image

def plot_ranking(logits, data_path, split_name, out_path='./plots/ranking.png'): 
    i2n = { v: k for k, v in json.load(open(data_path / 'mapping.txt')).items()}
    split  = np.array([l.split() for l in open(data_path / split_name)])
    # logits = torch.load(tensor_path)

    classes      = range(10)
    class_images = []
    for c in classes: 
        top    = logits[:,c].topk(5)
        top    = split[top.indices, 0]
        bottom = logits[:,c].topk(5, largest=False)
        bottom = split[bottom.indices, 0]

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



def tpr_plot(metadata_path, tensor_path, data_path, split_name): 
    true_class  = torch.tensor(np.array([int(l.split()[1]) for l in open(data_path / split_name)]))
    logits      = torch.load(tensor_path)
    pred_class  = logits.argmax(dim=3)
    correct     = (pred_class == true_class)
    tpr         = correct.sum(dim=2) / true_class.shape[0]

    plt.plot(tpr[1,:])
    plt.plot(tpr[0,:])

    models = [l for l in open(metadata_path)]
    for model_name, model_perf in zip(models, tpr): 
        plt.plot(model_perf, label="model")
        plt.ylabel("tpr")
        plt.xlabel("epoch")
    plt.legend()




def plot_tpr(model_tprs, out_path="./plots/tpr.png"): 
    for k, v in model_tprs.items(): 
        plt.plot(v, label=k)
    plt.ylabel("tpr")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(out_path)



def main():
    # logits = torch.randn((4050, 10))
    # torch.save(logits, 'test_tensor.pt')
    # path = Path('test_tensor.pt')
    # data_path = Path('./data/EuroSAT_RGB/')
    # split_name = Path('test.txt')
    # ranking_diagnostic(path, data_path, split_name)


    logits = torch.randn((3, 10, 4050, 10))
    torch.save(logits, 'val_tensor.pt')
    path = Path('val_tensor.pt')
    data_path = Path('./data/EuroSAT_RGB/')
    split = Path('val.txt')
    models = Path('./models.txt')
    tpr_plot(models, path, data_path, split)


if __name__ == "__main__":
    main()



import json
import torch
import numpy as np
from pathlib import Path 
import matplotlib.pyplot as plt

def ranking_diagnostic(tensor_path, data_path, split_name): 
    i2n = { v: k for k, v in json.load(open(data_path / 'mapping.txt')).items()}
    split  = np.array([l.split() for l in open(data_path / split_name)])
    logits = torch.load(tensor_path)

    class_images = []
    for c in [0, 1, 3]: 
        top    = logits[:,c].topk(5)
        top    = split[top.indices, 0]
        bottom = logits[:,c].topk(5, largest=False)
        bottom = split[bottom.indices, 0]

        class_images.append((
            [Image.open(data_path / b) for b in bottom],
            [Image.open(data_path / t) for t in top]
        ))

    plt.figure()
    for i, c in enumerate(class_images): 
        for k, img in enumerate(c[0], start=1): 
            plt.subplot(3, 10, k + 10*i)
            plt.imshow(img)
            plt.yticks([])
            plt.xticks([])
            if i == 0:
                plt.title("top {}".format(k))
            if k == 1: 
                plt.ylabel("{}".format(i2n[i]))

        for k, img in enumerate(c[1], start=1): 
            plt.subplot(3, 10, 5 + k + 10*i)
            plt.imshow(img)
            plt.yticks([])
            plt.xticks([])
            if i == 0:
                plt.title("bot {}".format(k))


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
        plt.plot(model_perf, label=model_name)
        plt.ylabel("tpr")
        plt.xlabel("epoch")
    plt.legend()


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



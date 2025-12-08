import numpy as np
import random 
from pathlib import Path 
import json


# CHANGE THIS IF THE DATA IS SOMEWHERE ELSE 
PATH  = Path("./data/EuroSAT_RGB/")


def verify_splits(path=Path("./data/EuroSAT_RGB/")): 
    with open(path / "val.txt", "r") as f: 
        val = set([l.split()[0] for l in f]) 

    with open(path / "train.txt", "r") as f: 
        train = set([l.split()[0] for l in f]) 

    with open(path / "test.txt", "r") as f: 
        test = set([l.split()[0] for l in f]) 

    print("val and train disjoint: {}"  .format(val.isdisjoint(train)))
    print("val and test disjoint: {}"   .format(val.isdisjoint(test)))
    print("train and test disjoint: {}" .format(train.isdisjoint(test)))
    print("total number of samples: {}" .format(len(val.union(train, test))))

    print("size of val: {}"  .format(len(val)))
    print("size of train: {}".format(len(train)))
    print("size of test: {}" .format(len(test)))



def data_prep(path: Path, split: tuple[float, float, float]): 
    mapping = dict()
    train = []
    test  = []
    val   = []
    label = 0
    for entry in path.iterdir(): 
        if entry.is_dir(): 
            mapping[entry.name] = label

            files     = [entry.name + '/' + p.name for p in entry.iterdir() if p.is_file()]
            random.shuffle(files)
            size      = len(files)
            train_end = int(round(split[0] * size))
            val_end   = train_end + int(round(split[1] * size))

            train.extend([(file, label) for file in files[:train_end]])
            test .extend([(file, label) for file in files[train_end:val_end]])
            val  .extend([(file, label) for file in files[val_end:]])

            label += 1


    json.dump(mapping, open(path / 'mapping.txt', 'w'))
    np.savetxt(path / "val.txt"  , np.array(val)  , fmt="%s")
    np.savetxt(path / "test.txt" , np.array(test) , fmt="%s")
    np.savetxt(path / "train.txt", np.array(train), fmt="%s")


def main(): 
    seed  = 42069
    random.seed(seed)
    np.random.seed(seed)

    split = (.7, .15, .15)
    data_prep(PATH, split)

    verify_splits(PATH)


if __name__ == "__main__":
    main()


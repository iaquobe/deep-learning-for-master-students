import torch 
from pathlib import Path 
from tqdm import tqdm

def train(model: torch.nn.Module, 
          loss_fn,
          optim: torch.optim.Optimizer,
          data: torch.utils.data.DataLoader,
          val: torch.utils.data.DataLoader,
          out_path: Path,
          epochs: int) -> list[float]: 
    '''
    trains a model

    Parameters: 
        model: the model to train
        optim: optimizer
        data: training data
        val: validation data
        out_path: where to save the best model out of all epochs
        epochs: the number of epochs to run

    Returns: 
        true positive rate across the epochs. This is used for plotting
    '''

    val_loss = float('inf')
    tpr_prog = []

    for _ in tqdm(range(epochs), desc='Epochs', unit='epochs'): 
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
                # break

        loss, acc, _ = test(model, loss_fn, val, desc="Validate")
        tpr_prog.append(acc)
        if loss < val_loss: 
            tqdm.write("\tnew best: writing to disk")
            val_loss = loss
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)

    return tpr_prog



def test(model: torch.nn.Module,
         loss_fn,
         data ,
         desc="Test") -> tuple[float, float, torch.Tensor]: 
    '''
    Validate or test a model. 

    Parameters: 
        model  : the model to use 
        loss_fn: the loss function
        data   : the validation/training data
        desc   : whether it is training or testing. This is for tqdm

    Returns: 
        loss: loss across the data set
        tpr: true positive rate across data set
        prediction logits: all logits for the dataset. 
            this is used for plotting the top-5 bottom-5 of each class
    '''

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

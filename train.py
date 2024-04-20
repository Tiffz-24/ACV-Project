import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import logging as log
from tqdm.notebook import tqdm

def train(model, epoch, dataloader, criterion, optimizer, scheduler=None):
    model.train()
    total, total_loss, total_correct = 0, 0., 0.

    tqdm_iterator = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (x, y) in tqdm_iterator:
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        output = model(x)
        prediction = torch.argmax(output, -1)
        loss = criterion(output, y.squeeze())
        total_loss += loss.item() * len(y)
        total_correct += (prediction == y.squeeze()).sum().item()
        total += len(y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        tqdm_iterator.set_postfix({
            "Loss": (total_loss / total), "Accuracy": (total_correct / total)
        })
    final_loss, final_acc = total_loss / total, total_correct / total
    log.info(
        "Reporting %.5f training loss, %.5f training accuracy for epoch %d." %
        (final_loss, final_acc, epoch)
    )
    return final_loss, final_acc, model
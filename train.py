import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import logging as log
from tqdm.notebook import tqdm

def train_one_epoch(model, epoch, dataloader, criterion, optimizer, scheduler):
    """
    Trains the given model for one epoch.

    Inputs:
        model: Torch.NN module
        epoch: int
        dataloader: Torch.Dataloader
        criterion: torch Criterion
        optimizer: torch Optimizer
        scheduler: torch Scheduler
    
    Outputs:
        final loss: float
        final_acc: float
        model: Torch.NN module
    """
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
    
def validate(model, dataloader, criterion):
    """
    Validate the model with the validation set

    Inputs:
        model: torch.nn
        dataloader: Dataloader
        criterion: torch Criterion
    Outputs: 
        avg_loss: flot
        avg_accuracy: float
    """
    model.eval()
    outputs = []
    labels = []
    total_loss, total_correct, total = 0, 0, 0
    
    with torch.no_grad():
        for x, y in dataloader:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            output = model(x)
            loss = criterion(output, y.squeeze())
            outputs.append(output.softmax(dim=-1)[:, 1])  # Assuming binary classification for AUROC
            labels.append(y.squeeze())
            
            prediction = torch.argmax(output, -1)
            total_loss += loss.item() * len(y)
            total_correct += (prediction == y.squeeze()).sum().item()
            total += len(y)
    
    avg_loss = total_loss / total
    avg_accuracy = total_correct / total
    outputs = torch.cat(outputs).cpu()
    labels = torch.cat(labels).cpu()
    return avg_loss, avg_accuracy, outputs, labels



def train(model, num_epochs, train_dataloader, val_dataloader, criterion, optimizer, scheduler=None):
    """
    Trains the given model.

    Inputs:
        model: Torch.NN module
        num_epochs: int
        dataloader: Torch.Dataloader
        criterion: torch Criterion
        optimizer: torch Optimizer
        scheduler: torch Scheduler
            Defaults to None
    
    Outputs:
        final loss: float
        final_acc: float
        model: Torch.NN module
    """
    print(scheduler)
    best_acc = 0.0  # Initialize the best accuracy
    best_model_wts = None  # To store the best model weights
    
    for epoch in range(num_epochs):
        log.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc, model = train_one_epoch(model, epoch, train_dataloader, criterion, optimizer, scheduler)
        log.info(f"Train Loss: {train_loss:.5f}, Train Accuracy: {train_acc:.5f}")
        
        # Validate
        val_loss, val_acc = validate(model, val_dataloader, criterion)
        log.info(f"Validation Loss: {val_loss:.5f}, Validation Accuracy: {val_acc:.5f}")
        
        # Check if the current epoch's validation accuracy is the best we've seen
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()  # Save the best model weights
            log.info(f"New best model found on validation data with accuracy: {best_acc:.5f}")
        
    # Load the best model weights back into the model
    model.load_state_dict(best_model_wts)
    log.info(f"Training complete. Best Validation Accuracy: {best_acc:.5f}")
    return model
from sklearn.metrics import roc_auc_score
from train import validate
def plot_auroc(model, val_dataloader, criterion):
    _, _, outputs, labels = validate(model, val_dataloader, criterion)
    auroc = roc_auc_score(labels, outputs)
    print("AUROC Score:", auroc)
    
    return auroc

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def plot_auroc_curve(model, val_dataloader, criterion):
    _, _, outputs, labels = validate(model, val_dataloader, criterion)
    fpr, tpr, _ = roc_curve(labels, outputs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(labels, outputs):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

from dataloading import load_train_data, get_dataloders
from train import train
from resnet import resnet18
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms

pcam_directory = '/Users/costanzasiniscalchi/Documents/Senior/ACV/project/histopathologic-cancer-detection'

train_transforms = transforms.Compose([
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet statistics
])
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet statistics
])


tr_ds, val_ds, tr_dl, val_dl = get_dataloders(pcam_directory, train_transforms, val_transforms) #using default batch size 32


model = resnet18(pretrained=True, num_classes=2)
if torch.cuda.is_available():
    model = model.cuda()
optimizer = torch.optim.AdamW([{'params': model.out_fc.parameters(), 'lr': 1e-3}])
criterion = nn.CrossEntropyLoss()
num_epochs = 10
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-2,
    total_steps=num_epochs * len(tr_dl),
)

num_epochs = 10

for i in range(num_epochs):
    train(model, i + 1, tr_dl, criterion, optimizer, scheduler)
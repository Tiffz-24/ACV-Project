from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from torch import nn
import torch.nn.functional as F

BATCH_SIZE = 64
def classifier(train_dataset, validation_dataset, test_dataset)
  #the dataset only looks at whether the center 32x32 region has metastatic tissue, so we want to crop the image accordingly
  def crop_images(image):
      square_length = 32
      return image.crop((square_length, square_length, square_length, square_length))

  for image in train_dataset:
    image = crop_images(image)

  for image in validation_dataset:
    image = crop_images(image)

  for image in test_dataset:
    image = crop_images(image)

  train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
  model = ResNet()
  print(training_and_testing(model, train_dataloader, validation_dataloader, test_dataloader))
  

def training_and_testing(model, train_loader, eval_loader, test_loader, num_epochs=50, lr=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        #validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in eval_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Save the model parameters

    print(f'Best epoch: {best_epoch}')
    print(f'Best validation loss: {best_val_loss}')

    # Load the best model parameters and evaluate on the test set
    model.eval()
    test_predictions = []
    test_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(test_labels, test_predictions)

    return test_accuracy

  
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding = 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding = 1)
        self.skip_connection = nn.Identity()
        if stride != 1 or in_channels != out_channels:
          #Use a Conv2d layer with kernel_size=1 to "resize" input
          self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        identity_x = self.skip_connection(x)
        identity_x = self.conv1(identity_x)
        identity_x = F.relu(identity_x)
        identity_x = self.conv2(identity_x)
        x = x + identity_x
        return x
      
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Need to crop the input to focus on the central 32x32 region
        # start off with 64 3x32x32 images
        # need to adjust the numbers for this dataset
        self.conv1 = nn.Conv2d(3, 32, 9) # 64x32x24x24
        self.pool1 = nn.MaxPool2d(2) # 64x32x12x12
        self.ResNet1 = ResNetBlock(32, 64, stride=2) # 64x64x12x12
        self.ResNet2 = ResNetBlock(64, 128) # 64x128x12x12

        self.conv2 = nn.Conv2d(128, 256, 5) # 64x256x8x8
        self.conv3 = nn.Conv2d(256, 256, 5) # 64x256x4x4
        self.pool2 = nn.MaxPool2d(2) # 64x256x2x2

        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(1024, 512)
        self.lin2 = nn.Linear(512, 100)
        self.lin3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.ResNet1(x)
        x = self.ResNet2(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return x

res_net = ResNet()


import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image

class PCamDataset(torch.utils.data.Dataset):
    def __init__(self, examples, transform=None):
        self.examples = examples
        self.transform = transform

    def __getitem__(self, index):
        image_fp, label = self.examples[index]
        image = Image.open(image_fp)
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.Tensor([label]).long()

    def __len__(self):
        return len(self.examples)

# def load_dataloaders(test_dataset, train_dataset, validation_dataset, batch_size = 64, num_workers = 4, shuffle = True):
    
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
#     validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
#     return test_dataloader, train_dataloader, validation_dataloader
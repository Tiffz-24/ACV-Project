import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image

class HEDataset(Dataset):
    def __init__(self, images_file_path, labels_file_path, transform=None, crop = None):
        self.images_file_path = images_file_path
        self.labels_file_path = labels_file_path
        self.transform = transform
        self.crop = crop
        
        # Load the file to check the shape of the dataset
        self._print_dataset_shapes()
    
    def _print_dataset_shapes(self):
        with h5py.File(self.images_file_path, 'r') as images_file:
            # Assuming you want to know the shape of the first image dataset
            
            first_image_key = list(images_file.keys())[0]
            image_shape = images_file[first_image_key].shape
            self.length = image_shape[0]
            if self.crop:
                print(f"Image shape {image_shape} will crop to ", [self.length, 3, self.crop, self.crop])
            else:
                print(f"Image shape {[self.length, 3, image_shape[2], image_shape[3]]}")
        with h5py.File(self.labels_file_path, 'r') as labels_file:
            # Assuming you want to know the shape of the first label dataset
            first_label_key = list(labels_file.keys())[0]
            label_shape = labels_file[first_label_key].shape

            if label_shape[0] != self.length:
                print("BAD: make error. Dataset x and y sizes do not match")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.images_file_path, 'r') as images_file:
            first_image_key = list(images_file.keys())[0]
            image_data = list(images_file[first_image_key])[idx]
            if image_data.shape[-1] == 3:
                image_data = image_data.transpose((2, 0, 1))  # RGB filter in 0 position

        with h5py.File(self.labels_file_path, 'r') as labels_file:
            first_label_key = list(labels_file.keys())[0]
            label_data = list(labels_file[first_label_key])[idx]
        print("got image and label data")
        if self.crop:
            w, h = image_data.shape[1], image_data.shape[2]
            startx = w // 2 - self.crop // 2
            starty = h // 2 - self.crop // 2
            image_data = image_data[:, starty:starty + self.crop, startx:startx + self.crop]
            print("cropped to ", image_data.shape)

        # Convert to PIL image using PIL Image module
        image_data = Image.fromarray(np.uint8(image_data.transpose(1, 2, 0)))

        if self.transform:
            image_data = self.transform(image_data)
            print("applied transform")

        return image_data, torch.tensor(label_data, dtype=torch.float32)

def load_datasets(test_x,test_y, train_x, train_y, valid_x,valid_y, transform=None, crop = None):

    test_dataset = HEDataset(test_x, test_y, transform, crop)
    train_dataset = HEDataset(train_x, train_y, transform, crop)
    validation_dataset = HEDataset(valid_x, valid_y, transform, crop)

    return test_dataset, train_dataset, validation_dataset

def load_dataloaders(test_dataset, train_dataset, validation_dataset, batch_size = 64, shuffle = True):
    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return test_dataloader, train_dataloader, validation_dataloader
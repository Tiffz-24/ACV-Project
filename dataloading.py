import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HEDataset(Dataset):
    def __init__(self, images_file_path, labels_file_path, transform=None):
        self.images_file_path = images_file_path
        self.labels_file_path = labels_file_path
        self.transform = transform
        
        # Load the file to check the shape of the dataset
        self._print_dataset_shapes()
    
    def _print_dataset_shapes(self):
        with h5py.File(self.images_file_path, 'r') as images_file:
            # Assuming you want to know the shape of the first image dataset
            first_image_key = list(images_file.keys())[0]
            image_shape = images_file[first_image_key].shape
            print(f"Image dataset shape: {image_shape}")
            self.length = image_shape[0]
        
        with h5py.File(self.labels_file_path, 'r') as labels_file:
            # Assuming you want to know the shape of the first label dataset
            first_label_key = list(labels_file.keys())[0]
            label_shape = labels_file[first_label_key].shape
            print(f"Label dataset shape: {label_shape}")

            if label_shape[0] != self.length:
                print("BAD: make error. Dataset x and y sizes do not match")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.images_file_path, 'r') as images_file:
            image_name = list(images_file.keys())[idx]
            image_data = images_file[image_name][()]

        with h5py.File(self.labels_file_path, 'r') as labels_file:
            label_data = labels_file[image_name][()]

        if self.transform:
            image_data = self.transform(image_data)

        return image_data, np.array(label_data, dtype=np.float32)

def load_datasets(test_x,test_y, train_x, train_y, valid_x,valid_y):

    test_dataset = HEDataset(test_x, test_y)
    train_dataset = HEDataset(train_x, train_y)
    validation_dataset = HEDataset(valid_x, valid_y)

    return test_dataset, train_dataset, validation_dataset

def load_dataloaders(test_dataset, train_dataset, validation_dataset, batch_size = 64, shuffle = True):
    BATCH_SIZE = batch_size
    
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    
    return test_dataloader, train_dataloader, validation_dataloader
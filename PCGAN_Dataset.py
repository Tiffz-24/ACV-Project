import torch
from torch.utils.data import Dataset
import os
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import csv


class PCGAN_Dataset(Dataset):
    def __init__(self, root_dir, label_mapping, transform=None, latent_dim=100):
        self.root_dir = root_dir
        self.label_mapping = label_mapping
        self.transform = transform
        self.latent_dim = latent_dim
        self.all_fps = [os.path.join(root_dir, fp) for fp in self.label_mapping.keys()]
        print(self.label_mapping, self.all_fps)

    def __getitem__(self, index):
        image_fp = self.all_fps[index]
        slide_id = os.path.basename(image_fp)  # Extracts filename from path
        image = Image.open(image_fp).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.label_mapping[slide_id]

        latent_vector = torch.randn(self.latent_dim)

        return image, latent_vector, label

    def __len__(self):
        return len(self.all_fps)

def get_dataset_dataloaders(pcam_directory, train_transforms, test_transforms, split = 0.6, batch_size = 128):
    label_mapping = {}
    with open(os.path.join(pcam_directory, 'train_labels.csv'), 'r') as f:
        reader = csv.reader(f)
        next(reader)  # To skip the header
        label_mapping = {slide_id +'.tif': int(label) for [slide_id, label] in reader}
    all_fps = [fp for fp in os.listdir(os.path.join(pcam_directory, 'train'))]
    for fp in all_fps: assert fp[-4:] == '.tif', fp[-4:]

    permutation = np.random.permutation(len(all_fps))
    num_train = int(len(permutation) * split)

    train_dataset = PCGAN_Dataset(os.path.join(pcam_directory, 'train'), {fp: label_mapping[fp] for fp in all_fps[:num_train] if fp in label_mapping}, transform=train_transforms)
    test_dataset = PCGAN_Dataset(os.path.join(pcam_directory, 'train'), {fp: label_mapping[fp] for fp in all_fps[num_train:] if fp in label_mapping}, transform=test_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_dataloader, test_dataloader
import torch
import numpy as np
from PIL import Image
import os
import csv
from tqdm.notebook import tqdm
from torch.utils.data import Dataset

class PCamDataset(Dataset):
    """
    Dataset class for PCam data
    """

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

def load_train_data(pcam_directory, split_train, split_test):
    """
    Splits train data into train/validation. Returns a dictionary mapping ids to labels and train and validation images.

    Inputs:
        pcam_directory: str
        split_train: float
            train split
        split_test: float
            test split
    Ouputs:
        label_mapping: dict
            Maps image ids to label
        train_fps: list
        val_fps: list
        test_fps: list
    """

    label_mapping = {}
    with open(os.path.join(pcam_directory, 'train_labels.csv'), 'r') as f:
        reader = csv.reader(f)
        next(reader)  # To skip the header
        label_mapping = {slide_id: int(label) for [slide_id, label] in reader}
    
    freq = {}
    for _, label in label_mapping.items():
        if label not in freq:
            freq[label] = 0.
        freq[label] += 1.
    freq = {k: v / len(label_mapping) for (k, v) in freq.items()}
    freq, len(label_mapping)

    all_fps = [fp for fp in os.listdir(os.path.join(pcam_directory, 'train'))]
    for fp in all_fps: assert fp[-4:] == '.tif', fp[-4:]

    permutation = np.random.permutation(range(len(all_fps)))
    print("split: train: ", int(len(permutation) * split_train), " val: ", int(len(permutation) * (split_test+split_train))-int(len(permutation) * split_train), "test: ", int(len(permutation))-int(len(permutation) * (split_test+split_train)))
    print("total: ", len(permutation))
    train_fps, val_fps, test_fps = (
        [
            (os.path.join(pcam_directory, 'train', all_fps[index]), label_mapping[all_fps[index][:-4]])
            for index in permutation[:int(len(permutation) * split_train)]
        ],
        [
            (os.path.join(pcam_directory, 'train', all_fps[index]), label_mapping[all_fps[index][:-4]])
            for index in permutation[int(len(permutation) * split_train):int(len(permutation) * (split_test+split_train))]
        ],
        [
            (os.path.join(pcam_directory, 'train', all_fps[index]), label_mapping[all_fps[index][:-4]])
            for index in permutation[int(len(permutation) * (split_test+split_train)):]
        ]
    )

    sizes = {}
    for fp, _ in tqdm(train_fps):
        sizes[Image.open(fp).size] = 1
    for fp, _ in tqdm(val_fps):
        sizes[Image.open(fp).size] = 1
    for fp, _ in tqdm(test_fps):
        sizes[Image.open(fp).size] = 1

    return label_mapping, train_fps, val_fps, test_fps

def get_dataloders(pcam_directory, train_transforms, val_transforms, test_transforms, batch_size = 32, split_train = 0.6, split_test = 0.2):
    """
    Loads data into dataset and dataloader objects for train and validation. Splits as given

    Inputs:
        pcam_directory: str
        train_transforms: torch Transform
        val_transforms: torch Transform
        batch_size: int
            Defaults to 32
        split: float
            train/test split. Defaults to 0.8
    Ouputs:
        tr_ds: PCAMDataset
        val_ds: PCamDataset
        tr_dl: torch Dataloader
        val_dl: torch Dataloader
    """
    label_mapping, train_fps, val_fps, test_fps = load_train_data(pcam_directory, split_train, split_test)
    tr_ds, val_ds, test_ds = PCamDataset(train_fps, transform=train_transforms), PCamDataset(val_fps, transform=val_transforms), PCamDataset(test_fps, transform=test_transforms)
    tr_dl, val_dl, test_dl = (
        torch.utils.data.DataLoader(tr_ds, batch_size, shuffle=True),
        torch.utils.data.DataLoader(val_ds, batch_size),
        torch.utils.data.DataLoader(test_ds, batch_size)
    )
    return label_mapping, tr_ds, val_ds, test_ds, tr_dl, val_dl, test_dl

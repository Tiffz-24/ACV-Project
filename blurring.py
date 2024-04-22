from PCamDataset import PCamDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision import transforms
from tqdm.notebook import tqdm
import matplotlib.patches as patches
import csv

device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pcam_directory = '/content/'
label_mapping = {}
with open(os.path.join(pcam_directory, 'train_labels.csv'), 'r') as f:
    reader = csv.reader(f)
    next(reader)  # To skip the header
    label_mapping = {slide_id: int(label) for [slide_id, label] in reader}
all_fps = [fp for fp in os.listdir(os.path.join(pcam_directory, 'train'))]
for fp in all_fps: assert fp[-4:] == '.tif', fp[-4:]

permutation = np.random.permutation(range(len(all_fps)))
dataset = PCamDataset([
        (os.path.join(pcam_directory, 'train', all_fps[index]), label_mapping[all_fps[index][:-4]])
        for index in permutation[:int(len(permutation) * .8)]
    ], transform= transforms.Compose([
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet statistics
    ]))

def show_dataset_images(dataset, indices, ncols=3):
    plt.figure(figsize=(15, 5))  # Adjust the size as needed
    for i, idx in enumerate(indices):
        image, _ = dataset[idx]
        if isinstance(image, torch.Tensor):  # Check if the image needs to be converted from a tensor
            image = image.permute(1, 2, 0).numpy()  # Adjust dimensions for Matplotlib
        plt.subplot(1, ncols, i + 1)
        plt.imshow(image)
        plt.title(f"Index: {idx}")
        plt.axis('off')
    plt.show()


def plot_with_rectangle(original_image, degraded_image, rect):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original image
    ax[0].imshow(original_image.permute(1, 2, 0).numpy())
    ax[0].set_title('Original Image')
    # Draw rectangle
    rect_patch = patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=1, edgecolor='r', facecolor='none')
    ax[0].add_patch(rect_patch)
    ax[0].axis('off')

    # Plot degraded image
    ax[1].imshow(degraded_image.permute(1, 2, 0).numpy())
    # Draw rectangle
    rect_patch = patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=1, edgecolor='r', facecolor='none')
    ax[1].add_patch(rect_patch)
    ax[1].set_title('Degraded Image with Red Rectangle')
    ax[1].axis('off')

    plt.show()
    

image, ___ = dataset.__getitem__(0)
blurred_image, rect = dataset.degrade_image(image.clone())
plot_with_rectangle(original_image=image, degraded_image=blurred_image, rect=rect)

import torch
import numpy as np
from PIL import Image
import os
import csv
from tqdm.notebook import tqdm
from torch.utils.data import Dataset
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms


def load_train_data(pcam_directory, split):
    pcam_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts PIL Image to a tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalizes the tensor
        ]
    )

    # download=True for first time to download pcam data
    pcam_train_data = torchvision.datasets.PCAM(
        root="./pcam", split="train", download=False, transform=pcam_transform
    )
    # pcam_val_data = torchvision.datasets.PCAM(
    #     root="./pcam", split="val", download=False
    # )
    # pcam_test_data = torchvision.datasets.PCAM(
    #     root="./pcam", split="test", download=False
    # )

    pcam_train_data_loader = torch.utils.data.DataLoader(
        pcam_train_data, batch_size=32, shuffle=True
    )
    print("Length of pcam_train_data_loader: ", len(pcam_train_data_loader))
    images, labels = next(iter(pcam_train_data_loader))
    return images, labels


def show_pcam_image(batch, num_imgs=4):
    plt.figure(figsize=(20, 10))
    for i in range(num_imgs):
        cur_img = batch["images"][i].permute(1, 2, 0)
        ax = plt.subplot(1, num_imgs, i + 1)
        plt.imshow(cur_img)
        ax.set_title(f"Label: {batch['labels'][i]}")
        ax.axis("off")
    plt.show()


def show_images(images, labels, num_images=6):
    images = images[:num_images]
    labels = labels[:num_images]
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
    for i, ax in enumerate(axes):
        img = images[i] / 2 + 0.5  # unnormalize the image
        img = img.numpy().transpose((1, 2, 0))
        ax.imshow(img)
        ax.set_title(f"Label: {labels[i]}")
        ax.axis("off")
    plt.show()


images, labels = load_train_data("./pcam", 0.8)
show_images(images, labels)

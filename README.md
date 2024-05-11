# ACV-Project

## Project Overview
Metastatic tissue clasificantion and generation using the PatchCamelyon (PCAM) dataset.


## Resnet/CNN

### Dataset
PCamDataset.py
dataloading.py

### Resnet 
resnet.py
test2.ipynb


## GAN Models
Pipleline for GAN models: <br>
<img width="385" alt="Screenshot 2024-05-10 at 11 25 17 PM" src="https://github.com/Tiffz-24/ACV-Project/assets/32231363/aac56685-3f40-4ec4-a006-4d43f9f916d1">

### Dataset
PCGAN_Dataset.py

### Preprocessing
deepfocus_and_blurring.ipynb
deepfocus.py

### Model 1: Conditional GAN
PCamGAN.py

### Model 2: Reconstructive Conditional GAN
reconstructGAN.ipynb

### Model 3: Transfer Learning using BigGAN
BigGAN_PCam.ipynb

### Group Responsibilities

Costanza
- Preprocessing: downloading and preprocessing dataset (checking classes, etc), dataset classes, data visualization
- Training: Conditional GAN, Reconstructive Conditional GAN (both run, but have issues with memory and GPU), some help with BigGAN
- Pipeline flowchart, test results tables
- Wrote explanations for the CGAN and RCGAN Methods, Analysis

Tiff:
- Preprocessing: gaussian blurring, cropping, etc. code, data visualization, DeepFocus testing code, also help with checking classes in preprocessing
- Training: CNNs, ResNets, beginner GAN (not included in final report due to not significant results)
- Wrote plotting code
- Introduction, Conclusion, Methods Overview

Cindy:
- BigGAN (not currently functioning)
- BigGAN methods
- ReadME file


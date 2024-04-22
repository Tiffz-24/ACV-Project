import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import utils as vutils
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.dropout(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Reduces any spatial dimensions to 1x1
            nn.Conv2d(128, 1, kernel_size=1),  # 1x1 convolution for final prediction
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(-1, 1)  # Flatten to [batch_size, 1]

# Function to generate noisy labels
def noisy_labels(device, true_size, fake_size, smooth_real=0.9, smooth_fake=0.1, flip_rate=0.05):
    """
    Returns lists of labels with added noise.
    Inputs:
        device: torch device
        true_size: int
        fake_size: int
        smooth_real: float
            Defaults to 0.9
        smooth_fake: float
            Defaults to 0.1
        flip_rate: float
            Defaults to 0.05
    """
    real_labels = smooth_real * torch.ones(true_size, 1, device=device)
    fake_labels = smooth_fake * torch.ones(fake_size, 1, device=device)

    # Introduce label flipping
    n_flips = int(min(true_size, fake_size) * flip_rate)
    real_labels[:n_flips] = smooth_fake  # Flipping real labels to fake
    fake_labels[:n_flips] = smooth_real  # Flipping fake labels to real

    # Shuffle to randomize which labels are flipped
    return real_labels[torch.randperm(true_size)], fake_labels[torch.randperm(fake_size)]


def train_gan(generator, discriminator, dataloader, device, num_epochs=50, save_path='gan_checkpoint.pth'):
    """
    Trains the GAN model
    Inputs:
        generator: Generator
        discriminator: Discriminator
        dataloader: PCamDataloader
        device: torch device
        num_epochs: int
            Defaults to 50
        save_path: str
            Defaults to gan_checkpoint.pth
    """
    # Loss function
    criterion = nn.BCELoss()

    # Optimizers
    lr_generator = 0.0004  # Increased learning rate for generator
    lr_discriminator = 0.00005  # Decreased learning rate for discriminator
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_generator, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_discriminator, betas=(0.5, 0.999))

    # Initialize for checkpointing
    best_g_loss = float('inf')

    # Fixed noise for visualizing progress
    fixed_noise = torch.randn(64, 3, 96, 96, device=device)  # Adjust to your input size

    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(dataloader):
            batch_size = images.size(0)
            real_images = images.to(device)

            # Get noisy labels for the batch
            real_labels, fake_labels = noisy_labels(batch_size, batch_size)

            ### Train Discriminator
            optimizer_D.zero_grad()

            # Real images
            outputs_real = discriminator(real_images)
            d_loss_real = criterion(outputs_real, real_labels)
            d_loss_real.backward()

            # Fake images
            noise = torch.randn(batch_size, 3, 96, 96, device=device)
            fake_images = generator(noise)
            outputs_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs_fake, fake_labels)
            d_loss_fake.backward()

            optimizer_D.step()
            d_loss = d_loss_real + d_loss_fake

            ### Train Generator
            optimizer_G.zero_grad()

            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()

            # Print losses and save generator images periodically
            if i % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

            # Save generated images to check progress
            if (i % 1000 == 0) or (i == len(dataloader) - 1):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                plt.figure(figsize=(10, 10))
                plt.axis("off")
                plt.title(f"Images at Epoch {epoch+1}, Step {i+1}")
                plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1, 2, 0)))
                plt.show()

            # Save best model
            if g_loss.item() < best_g_loss:
                best_g_loss = g_loss.item()
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                    'g_loss': g_loss,
                    'd_loss': d_loss,
                }, save_path)
                print(f'Checkpoint saved at Epoch {epoch+1}, Step {i+1} with G Loss: {g_loss.item()}')



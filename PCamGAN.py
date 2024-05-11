import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from os.path import join
from tqdm.notebook import tqdm
from dataclasses import dataclass


class CGanGenerator(nn.Module):
    def __init__(self, latent_dimension=100, n_classes=2, output_channels=3):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.n_classes = n_classes
        self.output_channels = output_channels
        self.embedding = nn.Embedding(n_classes, 50)
        self.dense = nn.Linear(50, 6 * 6)
        self.fc1 = nn.Linear(latent_dimension, 1024 * 6 * 6)
        self.conv_blocks = nn.ModuleList([
            self._get_conv_transpose_block(1025, 512),
            self._get_conv_transpose_block(512, 512),
            self._get_conv_transpose_block(512, 512),
            self._get_conv_transpose_block(512, output_channels, last_block=True)
        ])

    def _get_conv_transpose_block(self, in_channels, out_channels, last_block=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ]
        if last_block:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 5, stride=2, padding=2, output_padding=1),
                      nn.Tanh()]
        return nn.Sequential(*layers)

    def forward(self, noise, labels):
        # Ensure labels are of type torch.long
        labels = labels.long()  # Convert labels to long
        x = self.fc1(noise).view(-1, 1024, 6, 6)
        y = self.dense(self.embedding(labels)).view(-1, 1, 6, 6)
        x = torch.cat([x, y], 1)
        for block in self.conv_blocks:
            x = block(x)
        return x

class CGanDiscriminator(nn.Module):
    def __init__(self, input_shape=(3, 96, 96), n_classes=2):
        super().__init__()
        self.n_classes = n_classes
        self.conv_blocks = nn.ModuleList([
            self._get_conv_block(3, 64, first_block=True),
            self._get_conv_block(64, 128),
            self._get_conv_block(128, 256),
            self._get_conv_block(256, 512),
            self._get_conv_block(512, 512)
        ])
        # Calculate the output size of the last conv layer dynamically
        example_input = torch.rand(1, *input_shape)
        output_size = self.forward_features(example_input).view(-1).shape[0]
        print("Output size for linear layer:", output_size)  # Debug print

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_size, 1),
            nn.Sigmoid()
        )
        self.class_label = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_size, n_classes),
            nn.Softmax(dim=1)
        )

    def _get_conv_block(self, in_channels, out_channels, first_block=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5)
        ]
        if not first_block:
            layers.insert(1, nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        for block in self.conv_blocks:
            x = block(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        real_fake = self.classifier(x)
        class_label = self.class_label(x)
        return real_fake, class_label

class CGan(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, noise, labels):
        self.discriminator.eval()
        gen_imgs = self.generator(noise, labels)
        validity, _ = self.discriminator(gen_imgs)
        return validity

def get_transforms():
    return transforms.Compose([
        transforms.Resize((96, 96)),  # Assuming images need to be resized
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def generate_real_samples(batch, device='cpu'):
    """Prepares a batch of real samples from the DataLoader"""
    images, latent_vectors, labels = batch  # Adjust this line based on your DataLoader output
    images = images.to(device)
    labels = labels.to(device)
    y = torch.ones(images.size(0), 1, dtype=torch.float32).to(device)  # Labels for real samples are all ones
    return images, labels, y

def generate_latent_points(latent_dim, n_samples, n_classes=2, device='cpu'):
    """Generates a batch of latent vectors of random points"""
    z_input = torch.randn(n_samples, latent_dim, device=device)
    labels = torch.randint(0, n_classes, (n_samples,), device=device)
    return z_input, labels

def generate_fake_samples(generator, latent_dim, n_samples, device='cpu'):
    """Generates a batch of fake samples from latent vectors using the generator model."""
    z_input, labels = generate_latent_points(latent_dim, n_samples, device=device)
    with torch.no_grad():
        # Make sure to pass both z_input and labels to the generator
        images = generator(z_input, labels)
    y = torch.zeros(n_samples, 1, dtype=torch.float32).to(device)  # Labels for fake samples are all zeros
    return images, labels, y


@dataclass
class TrainParam:
    n_epochs: int
    batch_size: int
    latent_dim: int
    epoch_checkpoint: int
    n_summary_samples: int
    starting_epoch: int = 0
    output_path: str = './'  # Default output path
    model_path: str = './'   # Default model path

def trainer(gan_model, data_loader, train_param, device='cpu'):
    """Train GAN model using given parameters and return the trained model along with training history."""
    gan_model.to(device)
    gan_model.train()
    optimizer_d = torch.optim.Adam(gan_model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_g = torch.optim.Adam(gan_model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = torch.nn.BCELoss()
    history = {
        'train_d_loss': [],
        'train_g_loss': []
    }

    for epoch in range(train_param.starting_epoch, train_param.n_epochs):
        train_d_loss, train_g_loss = 0.0, 0.0
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1}/{train_param.n_epochs}", leave=False) as t:
            for batch in data_loader:
                X_real, labels_real, y_real = generate_real_samples(batch, device=device)

                # Training discriminator on real samples
                real_preds, _ = gan_model.discriminator(X_real)
                d_loss_real = criterion(real_preds, y_real)

                # Generate fake samples
                fake_images, labels_fake, y_fake = generate_fake_samples(gan_model.generator, train_param.latent_dim, X_real.size(0), device=device)
                fake_preds, _ = gan_model.discriminator(fake_images.detach())
                d_loss_fake = criterion(fake_preds, y_fake)

                # Update discriminator
                d_loss = d_loss_real + d_loss_fake
                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()

                # Update generator
                g_preds, _ = gan_model.discriminator(fake_images)
                g_loss = criterion(g_preds, torch.ones_like(g_preds))

                optimizer_g.zero_grad()
                g_loss.backward()
                optimizer_g.step()

                train_d_loss += d_loss.item()
                train_g_loss += g_loss.item()

                t.set_postfix(D_Loss=f"{d_loss.item():.4f}", G_Loss=f"{g_loss.item():.4f}")
                t.update(1)

        history['train_d_loss'].append(train_d_loss / len(data_loader))
        history['train_g_loss'].append(train_g_loss / len(data_loader))

        if (epoch+1) % train_param.epoch_checkpoint == 0:
            summarizer(epoch, gan_model, train_param.output_path, train_param.model_path, train_param, device)

        print(f'Epoch {epoch+1}, Avg D Loss: {train_d_loss / len(data_loader):.4f}, Avg G Loss: {train_g_loss / len(data_loader):.4f}')

    return gan_model, history



def summarizer(epoch, gan_model, output_path, model_path, train_param, device):
    """Generate and save images during training."""
    gan_model.eval()
    z_input, labels = generate_latent_points(train_param.latent_dim, train_param.n_summary_samples, device=device)
    with torch.no_grad():
        fake_images = gan_model.generator(z_input, labels)
    fake_images = ((fake_images * 0.5 + 0.5) * 255).cpu().numpy().astype(np.uint8)

    plot_images(fake_images, figsize=(15, 15), n_samples=train_param.n_summary_samples, epoch=epoch, output_path=output_path)
    torch.save(gan_model.generator.state_dict(), os.path.join(model_path, f'g_model_{epoch}.pt'))
    torch.save(gan_model.discriminator.state_dict(), os.path.join(model_path, f'd_model_{epoch}.pt'))
    gan_model.train()


def plot_images(X, figsize, n_samples, epoch, output_path=None):
    """Plot and save generated images."""
    plt.figure(figsize=figsize)
    sample_sqrt = int(sqrt(n_samples))
    plt.subplots_adjust(right=0.9, left=0.0, top=0.9, bottom=0.0, hspace=0.02, wspace=0.02)
    for i in range(n_samples):
        plt.subplot(sample_sqrt, sample_sqrt, 1 + i)
        plt.axis('off')
        plt.imshow(X[i].transpose(1, 2, 0))  # Transpose as needed depending on data format
    plt.show()
    if output_path:
        plt.savefig(join(output_path, f'generated_plot_{epoch}.png'))
        plt.close()

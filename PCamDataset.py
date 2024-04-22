
import random
from PIL import Image
import torch
from tqdm.notebook import tqdm

class PCamDataset(torch.utils.data.Dataset):
    def __init__(self, examples, transform=None):
        self.examples = examples
        self.transform = transform
        self.crop = 125
        self.kernel_cache = {}  # Cache kernels to avoid recalculating them

    def __getitem__(self, index):
        image_fp, label = self.examples[index]
        image = Image.open(image_fp)
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.Tensor([label]).long()

    def __len__(self):
        return len(self.examples)

    def crop_image(self, image_data):
        w, h = image_data.shape[1], image_data.shape[0]
        startx = w // 2 - self.crop // 2
        starty = h // 2 - self.crop // 2
        return image_data[starty:starty + self.crop, startx:startx + self.crop]

    def update_exclusion_list(self, exclude_indices):
        """ Update the list of indices to exclude from the dataset. """
        self.exclude_indices = set(exclude_indices)

    def degrade_all_images(self):
      degraded_images = []
      for image_fp, label in self.examples:
          image = Image.open(image_fp)
          if self.transform is not None:
              image = self.transform(image)
          degraded_image = self.degrade_image(image)
          degraded_images.append(degraded_image)
      return degraded_images

    def create_gaussian_kernel(self, size, sigma):
        """Creates a Gaussian kernel only if not already cached."""
        if (size, sigma) not in self.kernel_cache:
            ax = torch.linspace(-(size - 1) / 2., (size - 1) / 2., size)
            xx, yy = torch.meshgrid(ax, ax)
            kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            kernel = kernel / torch.sum(kernel)
            self.kernel_cache[(size, sigma)] = kernel.view(1, 1, size, size)
        return self.kernel_cache[(size, sigma)]

    def degrade_image(self, image):
        c, h, w = image.shape
        patch_size = random.randint(10, 32)
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        print(f"Patch size: {patch_size}, X: {x}, Y: {y}")

        # Extract the patch to work with
        patch = image[:, y:y + patch_size, x:x + patch_size]

        # Apply Gaussian blur to the patch
        if random.random() > 0.5:
            size = random.choice([3, 5, 7])  # Kernel size
            sigma = random.uniform(0.5, 1.5)  # Sigma for Gaussian kernel
            blur_kernel = self.create_gaussian_kernel(size, sigma).repeat(c, 1, 1, 1).to(image.device)
            patch = F.pad(patch, (size//2, size//2, size//2, size//2), mode='reflect')
            blurred_patch = F.conv2d(patch.unsqueeze(0), blur_kernel, padding=0, stride=1, groups=c).squeeze(0)

            # Place the blurred patch back into the original image
            image[:, y:y + patch_size, x:x + patch_size] = blurred_patch

        # Add Gaussian noise to the patch
        if random.random() > 0.5:
            noise = torch.randn_like(patch) * 0.05
            patch += noise  # In-place addition of noise

        rect = (x, y, patch_size, patch_size)
        return image, rect
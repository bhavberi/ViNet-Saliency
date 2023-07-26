import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Set the path to your folder containing the images
data_folder = "/path/to/your/images/folder"

# Define a transformation to convert images to tensors
transform = transforms.Compose([
    transforms.Resize((256, 256)),     # Resize the images (adjust the size as needed)
    transforms.ToTensor()              # Convert images to tensors
])

# Create a dataset from the folder
dataset = ImageFolder(data_folder, transform=transform)

# Create a data loader to iterate through the images
data_loader = DataLoader(dataset, batch_size=4, shuffle=False)  # Adjust batch_size as needed

# Initialize variables to store sum and sum of squares
sum_channels = np.zeros(3)  # RGB images have 3 channels
sum_channels_squared = np.zeros(3)

# Iterate through the data and calculate sum and sum of squares
for images, _ in data_loader:
    sum_channels += np.sum(images.numpy(), axis=(0, 2, 3))  # Sum over batch (0), height (2), and width (3) dimensions
    sum_channels_squared += np.sum(images.numpy() ** 2, axis=(0, 2, 3))

# Calculate the mean and std values
num_images = len(dataset)
mean = sum_channels / (num_images * 256 * 256)  # Assuming the images were resized to 256x256
std = np.sqrt((sum_channels_squared / (num_images * 256 * 256)) - mean ** 2)

print("Mean:", mean)
print("Std:", std)


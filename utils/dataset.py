import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage import exposure
from config import *

class ImageDataset(Dataset):
    def __init__(self, root_dir1,root_dir2):
        self.root_dir1 = root_dir1
        self.subdirectories1 = sorted(os.listdir(root_dir1))
        self.root_dir2 = root_dir2
        self.subdirectories2 = sorted(os.listdir(root_dir2))

    def __len__(self):
        return len(self.subdirectories1)

    def __getitem__(self, idx):
        subdir1 = self.subdirectories1[idx]
        subdir_path1 = os.path.join(self.root_dir1, subdir1)
        subdir2 = self.subdirectories2[idx]
        subdir_path2 = os.path.join(self.root_dir2, subdir2)
        #print('path1 ',subdir_path1)
        #print('path2',subdir_path2)
        images = []
        entropy = []  # To store magnitude spectra

        for filename in sorted(os.listdir(subdir_path1)):
            if filename.endswith(".png"):
                image_path = os.path.join(subdir_path1, filename)
                image = Image.open(image_path).convert('L')
                transform = transforms.Compose([transforms.Resize((128,128)),
                                                transforms.ToTensor()])
                image = transform(image)
                images.append(image)

        for filename in sorted(os.listdir(subdir_path2)):
            if filename.endswith(".png"):
                image_path = os.path.join(subdir_path2, filename)
                image = Image.open(image_path).convert('L')
                transform = transforms.Compose([transforms.Resize((128,128)),
                                                transforms.ToTensor()])
                image = transform(image)
                entropy.append(image)         # Compute the entropy spectrum
               

        # Ensure there are at least 17 images in the folder
        if len(images) < 17:
            raise ValueError(f"The folder '{subdir1}' must contain at least 17 images.")

        middle_index = len(images) // 2
        # Select 8 images to the left of the middle
        left_images = images[max(0, middle_index - 8):middle_index]
        # Select the middle image
        middle_image = images[middle_index]
        # Select 8 images to the right of the middle
        right_images = images[middle_index + 1:middle_index + 9]

        # Concatenate the three sets of images
        selected_images = left_images + [middle_image] + right_images
        selected_entropy = entropy[max(0, middle_index - 8):middle_index] + [entropy[middle_index]] + entropy[middle_index + 1:middle_index + 9]

        return torch.stack(selected_images), torch.stack(selected_entropy), int(subdir1)
        
dataset = ImageDataset(TRAIN_DIR1, TRAIN_DIR2)

# Assuming each subdirectory contains 15 images
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
#print(next(iter(train_loader)))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)



import os
import numpy as np
from PIL import Image
from skimage.filters.rank import entropy as skimage_entropy
from skimage.morphology import disk
import torch

device =  torch.device("cuda:0" )

# Function to calculate and save entropy map
def calculate_and_save_entropy(input_image_path, output_image_path, window_size=15):
    image = Image.open(input_image_path)
    gray_image = image.convert('L')
    gray_array = np.array(gray_image)

    # Calculate entropy for each pixel using skimage's entropy function
    entropy_map = skimage_entropy(gray_array, disk(window_size))

    # Normalize entropy values to range [0, 1]
    entropy_map_normalized = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min()) * 255

    # Convert array back to image and save
    entropy_image = Image.fromarray(entropy_map_normalized.astype(np.uint8))
    entropy_image.save(output_image_path)
input_root = '../Knee_png/'
output_root = '../Knee_png_entropy/'

# Iterate through each subfolder in the input directory
for root, dirs, files in os.walk(input_root):
    for dir_name in dirs:
        input_subfolder = os.path.join(input_root, dir_name)
        output_subfolder = os.path.join(output_root, dir_name)
        os.makedirs(output_subfolder, exist_ok=True)
        for filename in os.listdir(input_subfolder):
            input_image_path = os.path.join(input_subfolder, filename)
            output_image_path = os.path.join(output_subfolder, filename)
            calculate_and_save_entropy(input_image_path, output_image_path)


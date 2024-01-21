import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Define class-to-id mapping
cls2id = {"background": 0, "crop": 1, "weed": 2, "partial-crop": 3, "partial-weed": 4}

# Path to the folder containing PNG images and corresponding label images
labels_folder = r"C:\Users\juanm\Documents\IPCV_3\AdvancedMethodsIP\project-dataset\train\labels"
images_folder = r"C:\Users\juanm\Documents\IPCV_3\AdvancedMethodsIP\project-dataset\train\images"

# Function to calculate coverage percentages for each class in a single image
def calculate_coverage(image_path):
    image = np.array(Image.open(image_path))
    coverage = {cls: np.sum(image == cls_id) / image.size * 100 for cls, cls_id in cls2id.items()}
    return coverage

# Initialize lists to store coverage percentages for each image
all_coverages = []

# Loop over each image in the folder
for filename in os.listdir(labels_folder):
    if filename.endswith(".png"):
        label_image_path = os.path.join(labels_folder, filename)
        coverage = calculate_coverage(label_image_path)
        all_coverages.append((filename, coverage))

# Find images with maximum and minimum amounts of each class
max_images = {}
min_images = {}

for cls in cls2id:
    # Find image with maximum amount of the class
    max_images[cls] = max(all_coverages, key=lambda x: x[1][cls])[0]

    # Find image with minimum amount of the class
    min_images[cls] = min(all_coverages, key=lambda x: x[1][cls])[0]

# Display images with maximum and minimum amounts of each class
for cls in cls2id:
    max_image_path = os.path.join(images_folder, max_images[cls])
    min_image_path = os.path.join(images_folder, min_images[cls])

    # Display images using Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(Image.open(max_image_path))
    axes[0].set_title(f"Max {cls}")
    axes[0].axis("off")

    axes[1].imshow(Image.open(min_image_path))
    axes[1].set_title(f"Min {cls}")
    axes[1].axis("off")

    plt.show()

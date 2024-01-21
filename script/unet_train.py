import sys
import numpy as np
import random
import torch
from torchvision.transforms import Compose, PILToTensor
from torch.utils.data import DataLoader
# sys.path.append('C:/Users/budai/Downloads/am4ip-lab3-master_my/am4ip-lab3-master/src')
sys.path.append('/net/cremi/zbudai/am4ip-lab3-master-1/src')
import itertools
import matplotlib.pyplot as plt
from am4ip.dataset import CropSegmentationDataset
from am4ip.models import UNet
from am4ip.trainer_unet import BaselineTrainer
from am4ip.losses import CombinedLoss, DiceLoss
from am4ip.metrics_unet import EvaluateNetwork

from torchvision.transforms import Resize
# from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import Normalize
from torchvision import models 
# import segmentation_models_pytorch as smp
from torch import nn
import logging
import datetime
import matplotlib.patches as mpatches

def get_class_counts(num_classes):
    return [61674600, 8563795, 359237]
    class_counts = [0] * num_classes
    i =0
    for _, target in dataset:
        for class_index in range(dataset.get_class_number()):
            class_counts[class_index] += (target == class_index).sum().item()
        print(i)
        i+=1
    print("Class counts:", class_counts)
    return class_counts

# Get the current time and format it as a string
now = datetime.datetime.now()
filename = now.strftime('%Y-%m-%d_%H-%M-%S.log')

# Set up logging
logging.basicConfig(filename=filename, level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# ImageNet mean and standard deviation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

img_size = 256

transform = Compose([
    PILToTensor(),
    lambda x: x.float(),  # Convert tensor to float
    Resize((img_size, img_size), antialias=True),  # Resize  
    # Normalize(mean, std),  # Normalize with ImageNet mean and std
    lambda z: z.to(dtype=torch.float32)
])

target_transform = Compose([
    PILToTensor(),
    Resize((img_size, img_size), antialias=True),  # Resize  
    lambda z: z.to(dtype=torch.int64).squeeze(0)
])

val_transform = Compose([
    PILToTensor(),
    lambda x: x.float(),  # Convert tensor to float
    Resize((img_size, img_size), antialias=True),  # Resize
    # Normalize(mean, std),  # Normalize with ImageNet mean and std
    lambda z: z.to(dtype=torch.float32)
])

val_target_transform = Compose([
    PILToTensor(),
    Resize((img_size, img_size), antialias=True),  # Resize
    lambda z: z.to(dtype=torch.int64).squeeze(0)
])


dataset = CropSegmentationDataset(transform=transform, target_transform=target_transform, merge_small_items=True)
val_dataset = CropSegmentationDataset(set_type="val", transform=val_transform, target_transform=val_target_transform, merge_small_items=True)
# dataset.visualize_data()
num_classes = dataset.get_class_number()
print("Number of classes:", num_classes)

class_counts = get_class_counts(num_classes)

# Define the hyperparameters
batch_sizes = [16]
epoch_sizes = [10, 20] 
learning_rates = [0.01, 0.001]
# Create a list of all combinations of hyperparameters
grid_list = list(itertools.product(batch_sizes, epoch_sizes, learning_rates))

# Initialize variables to store the best parameters and the best score
best_params = None
best_score = float('-inf')

# Loop over all combinations of hyperparameters
logger.info("UNET, dice+bce loss, SGD optimizer")
for params in grid_list:
    logger.info(f"Parameters: batch_size={params[0]}, epoch={params[1]}, lr={params[2]}")
    batch_size, epoch, lr = params

    # Create the data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create the model, loss function, and optimizer
    # def __init__(self,in_channel,  n_classes, batchnorm=False,  bilinear=False):
    unet = UNet(in_channel=3, n_classes=num_classes)  # Create the U-Net model
    loss = CombinedLoss(class_counts=class_counts)
    optimizer = torch.optim.SGD(unet.parameters(), lr=lr, momentum=0.8)

    # Train the model
    trainer = BaselineTrainer(model=unet, loss=loss, optimizer=optimizer, use_cuda=True)  # Use the U-Net model
    trainer.fit(train_loader, val_data_loader=val_loader, epoch=epoch)

    # Evaluate the model
    unet_results, class_wise_results = EvaluateNetwork(unet, val_loader)  # Use the U-Net model
    # Visualize some figures

        # Define a color map
    colors = [
            [0, 0, 0],  # Class 0: Black
            [255, 0, 0],  # Class 1: Red
            [0, 255, 0],  # Class 2: Green
            [0, 0, 255],  # Class 3: Blue
            [255, 255, 0] # Class 4: Yellow 
            # Add more colors for more classes
        ]
    inputs, targets = next(iter(val_loader))
    colors = np.array(colors, dtype='uint8')

    # If the current score is better than the best score, update the best score and the best parameters
    if unet_results > best_score:
        best_score = unet_results
        best_params = params
        best_model = unet  # Save the best model
        torch.save(best_model.state_dict(), 'best_model_weights.pth')  # Save the best model's weightsams = params
    # Log the results
    logger.info(f"Score: {unet_results}\n  Class-wise results:")
    for i, result in enumerate(class_wise_results):
        logger.info(f"Class {i}: {result}")

logger.info(f"Best parameters: {best_params}\n Best score: {best_score}")


print("Best parameters:", best_params)
print("Best score:", best_score)

def visualize_predictions(model, num_samples=5):
    fig, axs = plt.subplots(num_samples, 4, figsize=(15, num_samples*3))
    all_labels = []

    for i in range(num_samples):
        # Select a random sample from the dataset
        idx = random.randint(0, len(val_dataset)-1)
        img, label = val_dataset[idx]

        # Convert PyTorch tensors to numpy arrays for visualization
        img_np = img.permute(1, 2, 0).numpy()
        label_np = label.numpy()

        # Normalize the image data to [0, 1] range
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        # Create a color map for the label
        cmap = plt.get_cmap('tab20b', len(val_dataset.id2cls))

        # Use the model to make a prediction
        with torch.no_grad():
            model.eval()
            pred = model(img.unsqueeze(0))
            pred = torch.argmax(pred, dim=1).squeeze().numpy()

        # Plot the image, label, prediction, and prediction projected onto the image
        axs[i, 0].imshow(img_np)
        axs[i, 0].set_title('Image')
        axs[i, 1].imshow(label_np, cmap=cmap, vmin=0, vmax=len(val_dataset.id2cls)-1)
        axs[i, 1].set_title('Label')
        axs[i, 2].imshow(pred, cmap=cmap, vmin=0, vmax=len(val_dataset.id2cls)-1)
        axs[i, 2].set_title('Prediction')
        axs[i, 3].imshow(img_np)
        axs[i, 3].imshow(pred, cmap=cmap, vmin=0, vmax=len(val_dataset.id2cls)-1, alpha=0.7)
        axs[i, 3].set_title('Prediction projected onto image')

        # Collect unique labels from all images
        all_labels.extend(np.unique(label_np))

    # Create legend
    unique_labels = np.unique(all_labels)
    legend_patches = [mpatches.Patch(color=cmap(i/(len(val_dataset.id2cls)-1)), label=val_dataset.id2cls[i]) for i in unique_labels]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Remove axis labels
    for ax in axs.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Load the best model
best_model = UNet(in_channel=3, n_classes=num_classes)
best_model.load_state_dict(torch.load('best_model_weights.pth'))
best_model.eval()

# Visualize predictions
visualize_predictions(best_model)





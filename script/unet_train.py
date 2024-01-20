import sys
import torch
from torchvision.transforms import Compose, PILToTensor
from torch.utils.data import DataLoader
# sys.path.append('C:/Users/budai/Downloads/am4ip-lab3-master_my/am4ip-lab3-master/src')
sys.path.append('/net/cremi/zbudai/am4ip-lab3-master-1/src')
import itertools

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
    Normalize(mean, std),  # Normalize with ImageNet mean and std
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
    Normalize(mean, std),  # Normalize with ImageNet mean and std
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
# Calculate class counts
# class_counts = [0] * dataset.get_class_number()
# for _, target in dataset:
#     for class_index in range(dataset.get_class_number()):
#         class_counts[class_index] += (target == class_index).sum().item()
# print("Class counts:", class_counts) 
class_counts = [61674600, 8563795, 359237]

# Define the hyperparameters
batch_sizes = [16,32]
epoch_sizes = [1, 10, 15, 20]
learning_rates = [0.001, 0.0001, 0.01]

# Create a list of all combinations of hyperparameters
grid_list = list(itertools.product(batch_sizes, epoch_sizes, learning_rates))

# Initialize variables to store the best parameters and the best score
best_params = None
best_score = float('-inf')

# Loop over all combinations of hyperparameters
logger.info("UNET")
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

    # If the current score is better than the best score, update the best score and the best parameters
    if unet_results > best_score:
        best_score = unet_results
        best_params = params
    # Log the results
    logger.info(f"Score: {unet_results}")
    logger.info(f"Class-wise results: ")
    for i, result in enumerate(class_wise_results):
        logger.info(f"Class {i}: {result}")
logger.info(f"Best parameters: {best_params}")
logger.info(f"Best score: {best_score}")

print("Best parameters:", best_params)
print("Best score:", best_score)
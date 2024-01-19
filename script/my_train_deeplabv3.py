import sys
import torch
from torchvision.transforms import Compose, PILToTensor
from torch.utils.data import DataLoader
# sys.path.append('C:/Users/budai/Downloads/am4ip-lab3-master_my/am4ip-lab3-master/src')
sys.path.append('/net/cremi/zbudai/am4ip-lab3-master-1/src')
import itertools

from am4ip.dataset import CropSegmentationDataset
from am4ip.models import SimpleNN, UNet
from am4ip.trainer import BaselineTrainer
from am4ip.losses import CombinedLoss, DiceLoss
from am4ip.metrics import nMAE, EvaluateNetwork

from torchvision.transforms import Resize
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import Normalize

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

img_size = 224

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

input_size = img_size*img_size*3

dataset = CropSegmentationDataset(transform=transform, target_transform=target_transform, merge_small_items=True)
val_dataset = CropSegmentationDataset(set_type="val", transform=val_transform, target_transform=val_target_transform, merge_small_items=True)
# dataset.visualize_data()
num_classes = dataset.get_class_number()
print("Number of classes:", num_classes)

# Define the hyperparameters
batch_sizes = [32, 64]
epoch_sizes = [5, 10, 15]
learning_rates = [0.1, 0.01, 0.005]

# Create a list of all combinations of hyperparameters
grid_list = list(itertools.product(batch_sizes, epoch_sizes, learning_rates))

# Initialize variables to store the best parameters and the best score
best_params = None
best_score = float('-inf')

# Loop over all combinations of hyperparameters
for params in grid_list:
    batch_size, epoch, lr = params

    # Create the data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create the model, loss function, and optimizer
    model = deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
    loss = DiceLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)

    # Train the model
    trainer = BaselineTrainer(model=model, loss=loss, optimizer=optimizer, use_cuda=False)
    trainer.fit(train_loader, val_data_loader=val_loader, epoch=epoch)

    # Evaluate the model
    deeplab_results = EvaluateNetwork(model, val_loader)

    # If the current score is better than the best score, update the best score and the best parameters
    if deeplab_results > best_score:
        best_score = deeplab_results
        best_params = params
    # Log the results
    logger.info(f"Parameters: {params}, Score: {deeplab_results}")

logger.info(f"Best parameters: {best_params}")
logger.info(f"Best score: {best_score}")

print("Best parameters:", best_params)
print("Best score:", best_score)
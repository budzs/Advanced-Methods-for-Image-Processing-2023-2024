import sys
import torch
from torchvision.transforms import Compose, PILToTensor
from torch.utils.data import DataLoader
# sys.path.append('C:/Users/budai/Downloads/am4ip-lab3-master_my/am4ip-lab3-master/src')
sys.path.append('/net/cremi/zbudai/am4ip-lab3-master-1/src')

from am4ip.dataset import CropSegmentationDataset
from am4ip.models import SimpleNN, UNet
from am4ip.trainer_unet import BaselineTrainer
from am4ip.losses import DiceLoss
from am4ip.metrics import nMAE, EvaluateNetwork

from torchvision.transforms import Resize
from sklearn.model_selection import ParameterGrid
import logging
import datetime
from torchvision.transforms import *

# Define the image transformations for augmentation
augmentations = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    RandomRotation(180),
   
])
# Define the label transformations for augmentation
label_augmentations = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    RandomRotation(180),
])

# Get the current time and format it as a string
now = datetime.datetime.now()
filename = now.strftime('%Y-%m-%d_%H-%M-%S_Unet.log')

# Set up logging
logging.basicConfig(filename=filename, level=logging.INFO, format='%(message)s')
logger = logging.getLogger()


img_size = 256
transform = Compose([
    augmentations,
    PILToTensor(),
    Resize((img_size, img_size), antialias=True),  # Resize  
    lambda z: z.to(dtype=torch.float32) / 127.5 - 1  # Normalize between -1 and 1
])

target_transform = Compose([
    label_augmentations,
    PILToTensor(),
    Resize((img_size, img_size), antialias=True),  # Resize  
    lambda z: z.to(dtype=torch.int64).squeeze(0)
])


dataset = CropSegmentationDataset(transform=transform, target_transform=target_transform)
val_dataset = CropSegmentationDataset(set_type="val", transform=transform, target_transform=target_transform)
# dataset.visualize_data()
num_classes = dataset.get_class_number()
print("Number of classes:", num_classes)

# Define the parameter grid
param_grid = {
    'lr': [1e-1, 1e-2, 1e-3],
    'epoch': [5, 10, 20],
    'batch_size': [32, 64],
    'img_size': [256, 512]
}

# Create the parameter grid
grid = ParameterGrid(param_grid)

# Iterate over each combination of parameters
for params in grid:
    lr = params['lr']
    epoch = params['epoch']
    batch_size = params['batch_size']
    img_size = params['img_size']

    # Update the transforms with the new img_size
    transform = Compose([
        augmentations,
        PILToTensor(),
        Resize((img_size, img_size), antialias=True),  # Resize  
        lambda z: z.to(dtype=torch.float32) / 127.5 - 1  # Normalize between -1 and 1
    ])

    target_transform = Compose([
        label_augmentations,    
        PILToTensor(),
        Resize((img_size, img_size), antialias=True),  # Resize  
        lambda z: z.to(dtype=torch.int64).squeeze(0)
    ])

    # Update the DataLoader with the new batch_size
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Update the optimizer with the new lr
    model = UNet(num_classes)
    loss = DiceLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Train the model with the new parameters
    trainer = BaselineTrainer(model=model, loss=loss, optimizer=optimizer, use_cuda=False)
    trainer.fit(train_loader,val_data_loader=val_loader, epoch=epoch)

    # Evaluate the model
    unet_results = EvaluateNetwork(model, val_loader)
    logger.info(f"UNet Results with lr={lr}, epoch={epoch}, batch_size={batch_size}, img_size={img_size}: {unet_results}")
    print(f"UNet Results with lr={lr}, epoch={epoch}, batch_size={batch_size}, img_size={img_size}: {unet_results}")
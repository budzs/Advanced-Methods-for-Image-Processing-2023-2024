import sys
import torch
from torchvision.transforms import Compose, PILToTensor
from torch.utils.data import DataLoader
# sys.path.append('C:/Users/budai/Downloads/am4ip-lab3-master_my/am4ip-lab3-master/src')
sys.path.append('/net/cremi/zbudai/am4ip-lab3-master-1/src')

from am4ip.dataset import CropSegmentationDataset
from am4ip.models import UNet
from am4ip.trainer_unet import BaselineTrainer
from am4ip.losses import DiceLoss, CombinedLoss
from am4ip.metrics import EvaluateNetwork

from torchvision.transforms import Resize
from sklearn.model_selection import ParameterGrid
import logging
import datetime
from torchvision.transforms import *

# Get the current time and format it as a string
now = datetime.datetime.now()
filename = now.strftime('%Y-%m-%d_%H-%M-%S_Unet.log')

# Set up logging
logging.basicConfig(filename=filename, level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Define the parameter grid
param_grid = {
    'lr': [1e-1, 1e-2, 5e-2],
    'epoch': [1, 5, 10, 20],
    'batch_size': [32, 64],
    'img_size': [256, 512,1024]
}

# Create the parameter grid
grid = ParameterGrid(param_grid)

# Iterate over each combination of parameters
for params in grid:
    lr = params['lr']
    epoch = params['epoch']
    batch_size = params['batch_size']
    img_size = params['img_size']
    logger.info(f"UNet with lr={lr}, epoch={epoch}, batch_size={batch_size}, img_size={img_size}")


    # Update the transforms with the new img_size
    transform = Compose([
        PILToTensor(),
        Resize((img_size, img_size), antialias=True),  # Resize  
        lambda z: z.to(dtype=torch.float32) / 127.5 - 1  # Normalize between -1 and 1
    ])

    target_transform = Compose([
        PILToTensor(),
        Resize((img_size, img_size), antialias=True),  # Resize  
        lambda z: z.to(dtype=torch.int64).squeeze(0)
    ])
    val_transform = Compose([
    PILToTensor(),
    Resize((img_size, img_size), antialias=True),  # Resize
    lambda z: z.to(dtype=torch.float32) / 127.5 - 1  # Normalize between -1 and 1
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

    # Update the DataLoader with the new batch_size
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Update the optimizer with the new lr
    model = UNet(num_classes)
    loss = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model with the new parameters
    trainer = BaselineTrainer(model=model, loss=loss, optimizer=optimizer, use_cuda=True)
    trainer.fit(train_loader,val_data_loader=val_loader, epoch=epoch)

    # Evaluate the model
    unet_results, class_wise_results = EvaluateNetwork(model, val_loader)
    logger.info(f"Score: {unet_results}")
    logger.info(f"Class-wise results: ")
    for i, result in enumerate(class_wise_results):
        logger.info(f"Class {i}: {result}")
    print(f"UNet Results: {unet_results}")
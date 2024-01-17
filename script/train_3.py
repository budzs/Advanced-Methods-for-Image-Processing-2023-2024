import sys
import torch
from torchvision.transforms import Compose, PILToTensor
from torch.utils.data import DataLoader
sys.path.append('C:/Users/budai/Downloads/am4ip-lab3-master_my/am4ip-lab3-master/src')

from am4ip.dataset import CropSegmentationDataset
from am4ip.models import SimpleNN, UNet
from am4ip.trainer import BaselineTrainer
from am4ip.losses import TotalLoss, DiceLoss
from am4ip.metrics import nMAE, EvaluateNetwork

from torchvision.transforms import Resize
img_size = 128
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
batch_size = 64
lr = 1e-3
epoch = 1
input_size = img_size*img_size*3
hidden_size = 1000

dataset = CropSegmentationDataset(transform=transform, target_transform=target_transform)
val_dataset = CropSegmentationDataset(set_type="val", transform=transform, target_transform=target_transform)
dataset.visualize_data()
num_classes = dataset.get_class_number()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the loss
loss = torch.nn.CrossEntropyLoss()
#loss = DiceLoss()

# Train CNN
# model = torch.nn.Sequential(torch.nn.Conv2d(3, 32, (3,3), padding="same"),
#                             torch.nn.Conv2d(32, dataset.get_class_number(), (3,3), padding="same"))
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# trainer = BaselineTrainer(model=model, loss=loss, optimizer=optimizer, use_cuda=False)
# trainer.fit(train_loader, epoch=epoch)
# cnn_results = EvaluateNetwork(model, test_loader)

# Train UNet
model = UNet(num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
trainer = BaselineTrainer(model=model, loss=loss, optimizer=optimizer, use_cuda=False)
trainer.fit(train_loader,val_data_loader=val_loader, epoch=epoch)
unet_results = EvaluateNetwork(model, val_loader)

# Compare results
# print("CNN Results:", cnn_results)
print("UNet Results:", unet_results)

print("job's done.")
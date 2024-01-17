
import sys
import torch
from torchvision.transforms import Compose, PILToTensor
from torch.utils.data import DataLoader
sys.path.append('C:/Users/budai/Downloads/am4ip-lab3-master_my/am4ip-lab3-master/src')
from am4ip.dataset import CropSegmentationDataset
from am4ip.trainer import BaselineTrainer
from am4ip.metrics import nMAE
from am4ip.losses import DiceLoss

from torchvision.transforms import Resize
img_size = 128
transform = Compose([
    PILToTensor(),
    Resize((img_size, img_size), antialias=True),  # Resize to 128x128
    lambda z: z.to(dtype=torch.float32) / 127.5 - 1  # Normalize between -1 and 1
])

target_transform = Compose([
    PILToTensor(),
    Resize((img_size, img_size), antialias=True),  # Resize to 128x128
    lambda z: z.to(dtype=torch.int64).squeeze(0)
])
batch_size = 64
lr = 1e-3
epoch = 10

dataset = CropSegmentationDataset(transform=transform, target_transform=target_transform)
val_dataset = CropSegmentationDataset(set_type="val", transform=transform, target_transform=target_transform)
dataset.visualize_data()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Implement VAE model:
# TODO: complete parameters and implement model forward pass + sampling
# model = CBDNetwork()
model = torch.nn.Sequential(torch.nn.Conv2d(3, 32, (3,3), padding="same"),
                            torch.nn.Conv2d(32, dataset.get_class_number(), (3,3), padding="same"))

# Implement loss function:
# TODO: implement the loss function as presented in the course
#loss = torch.nn.CrossEntropyLoss()  #TotalLoss()
loss = DiceLoss()
# Choose optimize:
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Implement the trainer
trainer = BaselineTrainer(model=model, loss=loss, optimizer=optimizer, use_cuda=False)

# Do the training
trainer.fit(train_loader, epoch=epoch)

# Compute metrics
# TODO: implement evaluation (compute IQ metrics on restaured images similarly to lab1)

print("job's done.")

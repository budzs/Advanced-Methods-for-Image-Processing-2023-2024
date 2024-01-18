import sys
import torch
from torchvision.transforms import Compose, PILToTensor
from torch.utils.data import DataLoader
sys.path.append('C:/Users/budai/Downloads/am4ip-lab3-master_my/am4ip-lab3-master/src')

from am4ip.dataset import CropSegmentationDataset
from am4ip.models import SimpleNN, UNet
from am4ip.trainer import BaselineTrainer
from am4ip.losses import CombinedLoss
from am4ip.metrics import nMAE, EvaluateNetwork

from torchvision.transforms import Resize
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import Normalize

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

# Rest of your code...
batch_size = 32
lr = 1e-2
epoch = 10
input_size = img_size*img_size*3
hidden_size = 1000

dataset = CropSegmentationDataset(transform=transform, target_transform=target_transform)
val_dataset = CropSegmentationDataset(set_type="val", transform=transform, target_transform=target_transform)
# dataset.visualize_data()
num_classes = dataset.get_class_number()
print("Number of classes:", num_classes)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the loss
#loss = torch.nn.CrossEntropyLoss()
loss = CombinedLoss()

# Train CNN
# model = torch.nn.Sequential(torch.nn.Conv2d(3, 32, (3,3), padding="same"),
#                             torch.nn.Conv2d(32, dataset.get_class_number(), (3,3), padding="same"))
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# trainer = BaselineTrainer(model=model, loss=loss, optimizer=optimizer, use_cuda=False)
# trainer.fit(train_loader, epoch=epoch)
# cnn_results = EvaluateNetwork(model, test_loader)



# Train DeepLabV3 with ResNet50 backbone
model = deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
trainer = BaselineTrainer(model=model, loss=loss, optimizer=optimizer, use_cuda=False)
trainer.fit(train_loader,val_data_loader=val_loader, epoch=epoch)
deeplab_results = EvaluateNetwork(model, val_loader)

print("DeepLabV3 Results:", deeplab_results)

print("job's done.")
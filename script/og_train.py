
import sys
import torch
from torchvision.transforms import Compose, PILToTensor
from torch.utils.data import DataLoader
sys.path.append('/net/cremi/zbudai/am4ip-lab3-master-1/src')
from am4ip.dataset import CropSegmentationDataset
from am4ip.trainer_unet import BaselineTrainer
from am4ip.metrics_unet import EvaluateNetwork

transform = Compose([PILToTensor(),
                     lambda z: z.to(dtype=torch.float32) / 127.5 - 1  # Normalize between -1 and 1
                     ])

target_transform = Compose([PILToTensor(),
                     lambda z: z.to(dtype=torch.int64).squeeze(0)
                     ])
batch_size = 32
lr = 1e-3
epoch = 1

dataset = CropSegmentationDataset(transform=transform, target_transform=target_transform)
val_dataset = CropSegmentationDataset(transform=transform, target_transform=target_transform, set_type="val")
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# Implement VAE model:
# TODO: complete parameters and implement model forward pass + sampling
# model = CBDNetwork()
model = torch.nn.Sequential(torch.nn.Conv2d(3, 32, (3,3), padding="same"),
                            torch.nn.Conv2d(32, dataset.get_class_number(), (3,3), padding="same"))

# Implement loss function:
# TODO: implement the loss function as presented in the course
loss = torch.nn.CrossEntropyLoss()  #TotalLoss()

# Choose optimize:
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Implement the trainer
trainer = BaselineTrainer(model=model, loss=loss, optimizer=optimizer, use_cuda=False)

# Do the training
trainer.fit(train_loader,val_data_loader=val_loader, epoch=epoch)

# Compute metrics
# TODO: implement evaluation (compute IQ metrics on restaured images similarly to lab1)
mean_iou, mean_ious = EvaluateNetwork(model, val_loader)
print("Mean IoU: ", mean_iou)
print("Mean IoUs: ")
for i in range(len(mean_ious)):
    print(f'Class{i} :{ mean_ious[i]}')
print("job's done.")

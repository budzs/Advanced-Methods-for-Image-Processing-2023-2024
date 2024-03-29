# Crop and Weed Segmentation Project 🌱🌾

[![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/downloads/release)

## Context 🌍
This project involves the segmentation of crops and weeds from images taken by a drone over crop fields. The task is to segment each pixel into either the background, a crop, or a weed. A dataset is provided, composed of a set of training images and masks (around 1400 images), a validation set of images and masks (around 450), and a testing set of images only (around 350 images). Additionally, a class to load the data is provided.

The data is available at the following location:
/net/ens/am4ip/datasets/project-dataset
Please change if you want to use.

## Description 🔍
In this project, I have experimented with various loss functions and models to optimize the performance of the semantic segmentation task.

For the loss functions, I have tried both Binary Cross Entropy (BCE) loss and Dice loss. BCE loss is a popular choice for binary classification problems, while Dice loss is often used for segmentation tasks as it is more sensitive to the overlap between the prediction and ground truth. I have also tried combining these two loss functions and adjusting their weights to see if a balance between them could improve the performance. The CombinedLoss class in src/am4ip/losses.py implements this combination.

For the evaluation metric, I have used Intersection over Union (IoU), which is a common metric for measuring the accuracy of an object detector on a particular dataset. The EvaluateNetwork class in src/am4ip/metrics.py calculates this metric.

In terms of models, I have tried UNet, DeepLabv3, and FCN-ResNet50. UNet is a popular choice for biomedical image segmentation due to its good performance in terms of accuracy. DeepLabv3 and FCN-ResNet50 are both state-of-the-art models for semantic image segmentation, with DeepLabv3 being known for its encoder-decoder structure with atrous convolutions, and FCN-ResNet50 for its fully convolutional nature. The implementations of these models can be found in the UNet class in src/am4ip/models.py.

Through these experiments, I aim to find the best combination of loss function and model for this specific task.

## Getting Started

### Prerequisites

List any prerequisites or requirements for running the code.

### Downloading Weights

To use the pre-trained weights for image classification, follow these steps:

1. Download the weights from [this link](https://drive.google.com/drive/folders/181eg_U2ldNvJEY8FR97I3OfndaJ6zDkh?usp=sharing).

    ```bash
    wget https://jm-pt.eu/wp-content/uploads/2024/01/weights.zip
    ```

    or

    ```bash
    curl -O https://jm-pt.eu/wp-content/uploads/2024/01/weights.zip
    ```

2. Unzip the downloaded file.

    ```bash
    unzip weights.zip
    ```

### Running the Code

Now, you can load the pre-trained weights (`model.pth`) and use them for image classification. Include code snippets or instructions for running the code.

```python
import torch
from torchvision.transforms import Compose, PILToTensor, Resize
from torch.utils.data import DataLoader
from am4ip.dataset import CropSegmentationDataset
from am4ip.models import UNet
from am4ip.metrics_unet import EvaluateNetwork

img_size = 256

transform = Compose([
    PILToTensor(),
    lambda x: x.float(),  # Convert tensor to float
    Resize((img_size, img_size), antialias=True),  # Resize  
    lambda z: z.to(dtype=torch.float32)
])

target_transform = Compose([
    PILToTensor(),
    Resize((img_size, img_size), antialias=True),  # Resize  
    lambda z: z.to(dtype=torch.int64).squeeze(0)
])

# Load the test dataset
test_dataset = CropSegmentationDataset(set_type="test", transform=transform, target_transform=target_transform, merge_small_items=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the best model
num_classes = test_dataset.get_class_number()
best_model = UNet(in_channel=3, n_classes=num_classes)
best_model.load_state_dict(torch.load('model.pth'))
best_model.eval()

# Evaluate the model on the test set
unet_results, class_wise_results = EvaluateNetwork(best_model, test_loader)

print(f"Score: {unet_results}\n  Class-wise results:")
for i, result in enumerate(class_wise_results):
    print(f"Class {i}: {result}")

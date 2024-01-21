## Crop and Weed Segmentation Project
# Context
This project involves the segmentation of crops and weeds from images taken by a drone over crop fields. The task is to segment each pixel into either the background, a crop, or a weed. A dataset is provided, composed of a set of training images and masks (around 1400 images), a validation set of images and masks (around 450), and a testing set of images only (around 350 images). Additionally, a class to load the data is provided.

The data is available at the following location:
/net/ens/am4ip/datasets/project-dataset
Please change if you want to use.
# Description
In this project, I have experimented with various loss functions and models to optimize the performance of the semantic segmentation task.

For the loss functions, I have tried both Binary Cross Entropy (BCE) loss and Dice loss. BCE loss is a popular choice for binary classification problems, while Dice loss is often used for segmentation tasks as it is more sensitive to the overlap between the prediction and ground truth. I have also tried combining these two loss functions and adjusting their weights to see if a balance between them could improve the performance. The CombinedLoss class in src/am4ip/losses.py implements this combination.

For the evaluation metric, I have used Intersection over Union (IoU), which is a common metric for measuring the accuracy of an object detector on a particular dataset. The EvaluateNetwork class in src/am4ip/metrics.py calculates this metric.

In terms of models, I have tried UNet, DeepLabv3, and FCN-ResNet50. UNet is a popular choice for biomedical image segmentation due to its good performance in terms of accuracy. DeepLabv3 and FCN-ResNet50 are both state-of-the-art models for semantic image segmentation, with DeepLabv3 being known for its encoder-decoder structure with atrous convolutions, and FCN-ResNet50 for its fully convolutional nature. The implementations of these models can be found in the UNet class in src/am4ip/models.py.

Through these experiments, I aim to find the best combination of loss function and model for this specific task.

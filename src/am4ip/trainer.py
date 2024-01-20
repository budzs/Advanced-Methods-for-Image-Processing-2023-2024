from typing import Callable, List
import torch
import torch.utils.data as data
import logging
import datetime

# Get the current time and format it as a string
now = datetime.datetime.now()
filename = 'training_' + now.strftime('%Y-%m-%d_%H-%M-%S') + '_trainer.log'

# Set up logging
logging.basicConfig(filename=filename, level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

class BaselineTrainer:
    def __init__(self, model: torch.nn.Module,
                 loss: Callable,
                 optimizer: torch.optim.Optimizer,
                 use_cuda=True):
        self.model = model
        self.loss = loss
        self.use_cuda = use_cuda
        self.optimizer = optimizer

        if self.use_cuda:
            self.model = self.model.to(device="cuda:0")
            print("CUDA is available")
        else:
            print("CUDA is not available")

    def fit(self, train_data_loader: data.DataLoader, val_data_loader: data.DataLoader, epoch: int):
        num_classes = train_data_loader.dataset.get_class_number()
        for e in range(epoch):
            # Training phase
            self.model.train()
            avg_loss = 0.
            n_batch = 0
            logger.info(f"Start epoch {e+1}/{epoch}")

            print(f"Start epoch {e+1}/{epoch}")
            for i, (input_img, label) in enumerate(train_data_loader):
                # Reset previous gradients
                self.optimizer.zero_grad()

                # Move data to cuda is necessary:
                if self.use_cuda:
                    input_img = input_img.cuda()
                    label = label.cuda()

                # Make forward
                # TODO change this part to fit your loss function
                output = self.model(input_img)
                output_tensor = output['out']
                loss = self.loss(output_tensor, label)
                loss.backward()

                # Adjust learning weights
                self.optimizer.step()
                avg_loss += loss.item()
                n_batch += 1

                print(f"{i+1}/{len(train_data_loader)}: loss = {avg_loss / n_batch}")
            print()

            # Validation phase
            self.model.eval()
            ious = []
            with torch.no_grad():
                for input_img, label in val_data_loader:
                    if self.use_cuda:
                        input_img = input_img.cuda()
                        label = label.cuda()

                    outputs = self.model(input_img)
                    output_tensor = outputs['out']  # replace 'out' with the actual key

                    _, predicted = torch.max(output_tensor.data, 1)

                    iou = IOU(predicted, label, num_classes) 
                    ious.append(iou)

            mean_iou = sum(ious) / len(ious)
            logger.info(f" loss = {avg_loss / n_batch}")
            logger.info(f'Epoch {e+1}, Validation mean IoU: {mean_iou}')

            print(f'Epoch {e+1}, Validation mean IoU: {mean_iou}')

        return avg_loss
    
def IOU(annotation, prediction, num_classes):
    ious = []
    for i in range(num_classes):
        truth = (annotation == i)
        pred = (prediction == i)
        intersection = torch.sum(torch.logical_and(truth, pred))
        union = torch.sum(torch.logical_or(truth, pred))
        epsilon = 1e-7  # Small constant to avoid division by zero
        iou = intersection.float() / (union.float() + epsilon)
        ious.append(iou)

    return sum(ious) / len(ious)  # return mean IoU

    return sum(ious) / len(ious)  # return mean IoU
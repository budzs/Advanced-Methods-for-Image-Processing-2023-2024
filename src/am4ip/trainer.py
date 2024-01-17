from typing import Callable, List
import torch
import torch.utils.data as data

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
            self.model = self.model.to(device="cuda:1")
            print("CUDA is available")
        else:
            print("CUDA is not available")

    def fit(self, train_data_loader: data.DataLoader, val_data_loader: data.DataLoader, epoch: int):
        for e in range(epoch):
            # Training phase
            self.model.train()
            avg_loss = 0.
            n_batch = 0
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
                loss = self.loss(self.model(input_img), label)
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
                    _, predicted = torch.max(outputs.data, 1)

                    iou = IOU(predicted, label) 
                    ious.append(iou)

            mean_iou = sum(ious) / len(ious)
            print(f'Epoch {e+1}, Validation mean IoU: {mean_iou}')

        return avg_loss
    
def IOU(annotation, prediction):
    ious = []
    for i in range(5):
        truth = annotation[i,:,:]
        pred = prediction[i,:,:]
        both = truth + pred
        ones = torch.ones_like(both)
        intersection = ones[both == 2]
        union = ones[both > 0]
        epsilon = 1e-7  # Small constant to avoid division by zero
        iou = sum(intersection) / (sum(union) + epsilon)
        ious.append(iou)

    return sum(ious) / len(ious)  # return mean IoU

import torch
from abc import ABC, abstractmethod


class IQMetric(ABC):
    """Abstract IQ metric class.
    """
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class FullReferenceIQMetric(IQMetric):
    """Abstract class to implement full-reference IQ metrics.
    """

    @abstractmethod
    def __call__(self, im: torch.Tensor, im_ref: torch.Tensor, *args) -> torch.Tensor:
        """Compute the metric over im and

        :param im: Batch of distorted images. Size = N x C x H x W
        :param im_ref: Batch of reference images. Size = N x C x H x W
        :return: IQ metric for each pair. Size = N
        """
        raise NotImplementedError


class NoReferenceIQMetric(IQMetric):
    """Abstract class to implement no-reference IQ metrics.
    """

    @abstractmethod
    def __call__(self, im: torch.Tensor, *args) -> torch.Tensor:
        """Compute the metric over im and

        :param im: Batch of distorted images. Size = N x C x H x W
        :return: IQ metric for each pair. Size = N
        """
        raise NotImplementedError


class NormalizedMeanAbsoluteError(FullReferenceIQMetric):
    """Compute normalized mean absolute error (MAE) on images.

    Note that nMAE is a distortion metric, not a quality metric. This means that it should be negatively
    correlated with Mean Opinion Scores.
    """
    def __init__(self, norm=255.):
        super(NormalizedMeanAbsoluteError, self).__init__(name="nMAE")
        self.norm = norm

    def __call__(self, im: torch.Tensor, im_ref: torch.Tensor, *args) -> torch.Tensor:
        return torch.mean(torch.abs(im - im_ref) / self.norm, dim=[1, 2, 3])  # Average over C x H x W


class PSNR(FullReferenceIQMetric):
    """Compute Peak Signal-to-Noise Ratio (PSNR) on images.
    """
    def __init__(self, max_val=255.):
        super(PSNR, self).__init__(name="PSNR")
        self.max_val = max_val

    def __call__(self, im: torch.Tensor, im_ref: torch.Tensor, *args) -> torch.Tensor:
        mse = torch.mean((im - im_ref) ** 2, dim=[1, 2, 3])  # Mean Squared Error over C x H x W
        return 20 * torch.log10(self.max_val / torch.sqrt(mse))  # PSNR
    

class MSE(NoReferenceIQMetric):
    """Compute Mean Squared Error (MSE) on images.
    """
    def __init__(self):
        super(MSE, self).__init__(name="MSE")

    def __call__(self, im: torch.Tensor, *args) -> torch.Tensor:
        return torch.mean(im ** 2, dim=[1, 2, 3])  # Mean Squared Error over C x H x W
    
# Aliases
nMAE = NormalizedMeanAbsoluteError


def IOU(annotation, prediction, num_classes=5):
    ious = []
    for i in range(num_classes):
        truth = (annotation == i)
        pred = (prediction == i)
        intersection = torch.logical_and(truth, pred)
        union = torch.logical_or(truth, pred)
        iou = intersection.float().sum() / (union.float().sum() + 1e-7)  # Add a small constant to avoid division by zero
        ious.append(iou.item())

    return ious

def EvaluateNetwork(model, test_loader):
    num_classes = test_loader.dataset.get_class_number()
    id2cls = {0: "background",
              1: "crop",
              2: "weed",
              3: "partial-crop",
              4: "partial-weed"}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    ious = [0.0] * num_classes  # Initialize IoUs for each class
    total = [0] * num_classes  # Initialize total count for each class

    with torch.no_grad():
        for batch in test_loader:
            img, target = batch
            img = img.to(device)

            output = model(img)
            preds = output.argmax(dim=1).cpu()

            target = target.cpu()
            # Calculate IoU for each class
            iou_per_class = IOU(target, preds, num_classes=num_classes)

            for c in range(num_classes):
                ious[c] += iou_per_class[c]
                total[c] += 1

    # Calculate mean IoU for each class
    mean_ious = [ious[c] / total[c] if total[c] > 0 else 0 for c in range(num_classes
                                                                          )]
    # Print IoUs for each class
    for c in range(num_classes):
        print(f"Class {id2cls[c]} IoU: {mean_ious[c]:.6f}")

    # Calculate and print mean IoU across all classes
    mean_iou = sum(mean_ious) / len(mean_ious)
    print(f"Mean IoU: {mean_iou:.6f}")

    return mean_iou, mean_ious
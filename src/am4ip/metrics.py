
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


def IOU(annotation, prediction):
    ious = []
    for i in range(3):
        truth = annotation[i,:,:]
        pred = prediction[i,:,:]
        both = truth + pred
        ones = torch.ones_like(both)
        intersection = ones[both == 2]
        union = ones[both > 0]
        iou = sum(intersection) / sum(union)
        ious.append(iou)

    return ious
def EvaluateNetwork(model, test_loader):
    id2cls = {0: "background",
              1: "crop",
              2: "weed",
              3: "partial-crop",
              4: "partial-weed"}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    accuracies = [0.0] * 5  # Initialize accuracies for each class
    total = [0] * 5  # Initialize total count for each class
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for batch in test_loader:
            img, target = batch
            img = img.to(device)

            output = model(img)
            preds = output.argmax(dim=1).cpu()

            for c in range(5):
                correct_pixels = (preds == c) & (target == c)
                total_pixels_class = (target == c).sum().item()
                accuracies[c] += correct_pixels.sum().item()
                total[c] += total_pixels_class

                total_correct += correct_pixels.sum().item()
                total_pixels += total_pixels_class

    # Calculate mean accuracy for each class
    mean_accuracies = [accuracies[c] / total[c] if total[c] > 0 else 0 for c in range(5)]

    # Calculate overall accuracy
    overall_accuracy = total_correct / total_pixels

    # Print accuracies for each class
    for c in range(5):
        print(f"Class {id2cls[c]} accuracy: {round(mean_accuracies[c], 6)}")

    print(f"Overall accuracy: {round(overall_accuracy, 6)}")

    return mean_accuracies, overall_accuracy


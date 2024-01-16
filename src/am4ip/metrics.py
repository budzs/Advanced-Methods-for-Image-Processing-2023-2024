
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

def EvaluateNetwork(net, testloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    net.to(device)
    net.eval()
    criterion = torch.nn.MSELoss()
    #compute mean test loss and iou
    sum_weed_iou = 0.0
    sum_crop_iou = 0.0
    sum_soil_iou = 0.0

    sum_loss = 0.0
    for i, batch in enumerate(testloader):
        print("working on image " + str(i))
        img = batch['image'].to(device)
        annot = batch['annotation'].to(device)
        output = net(img)['out']
        # compute MSE Loss
        loss = criterion(output, annot)

        # construct prediction and compute per class IOU
        output = output.squeeze() # single image assuming batch size = 1
        classes = torch.argmax(output, dim=0)
        r = torch.zeros_like(classes)
        g = torch.zeros_like(classes)
        b = torch.zeros_like(classes)

        idx = classes == 0
        r[idx] = 1
        idx = classes == 1
        g[idx] = 1
        idx = classes == 2
        b[idx] = 1

        prediction = torch.stack([r, g, b], axis=0).float()
        ious = IOU(prediction, annot[0])

        sum_loss += loss.item()
        sum_weed_iou += ious[0]
        sum_crop_iou += ious[1]
        sum_soil_iou += ious[2]

    print("loss: " + str(sum_loss / len(testloader)))
    print("weed iou: " + str(sum_weed_iou/ len(testloader)))
    print("crop iou: " + str(sum_crop_iou/ len(testloader)))
    print("soil iou: " + str(sum_soil_iou/ len(testloader)))



# Aliases
nMAE = NormalizedMeanAbsoluteError

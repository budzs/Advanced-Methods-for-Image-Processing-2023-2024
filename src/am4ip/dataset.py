
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable
import matplotlib.pyplot as plt
import numpy as np
import random

from .utils import expanded_join


class CropSegmentationDataset(Dataset):
    ROOT_PATH: str = "/net/ens/am4ip/datasets/project-dataset"
    id2cls: dict = {0: "background",
                    1: "crop",
                    2: "weed",
                    3: "partial-crop",
                    4: "partial-weed"}
    cls2id: dict = {"background": 0,
                    "crop": 1,
                    "weed": 2,
                    "partial-crop": 3,
                    "partial-weed": 4}

    def __init__(self, set_type: str = "train", transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 merge_small_items: bool = False,
                 remove_small_items: bool = False):
        """Class to load datasets for the Project.

        Remark: `target_transform` is applied before merging items (this eases data augmentation).

        :param set_type: Define if you load training, validation or testing sets. Should be either "train", "val" or "test".
        :param transform: Callable to be applied on inputs.
        :param target_transform: Callable to be applied on labels.
        :param merge_small_items: Boolean to either merge classes of small or occluded objects.
        :param remove_small_items: Boolean to consider as background class small or occluded objects. If `merge_small_items` is set to `True`, then this parameter is ignored.
        """
        super(CropSegmentationDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.merge_small_items = merge_small_items
        self.remove_small_items = remove_small_items

        if set_type not in ["train", "val", "test"]:
            raise ValueError("'set_type has an unknown value. "
                             f"Got '{set_type}' but expected something in ['train', 'val', 'test'].")

        self.set_type = set_type
        images = glob(expanded_join(self.ROOT_PATH, set_type, "images/*"))
        images.sort()
        self.images = np.array(images)

        labels = glob(expanded_join(self.ROOT_PATH, set_type, "labels/*"))
        labels.sort()
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        input_img = Image.open(self.images[index], "r")
        target = Image.open(self.labels[index], "r")

        if self.transform is not None:
            input_img = self.transform(input_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.merge_small_items:
            target[target == self.cls2id["partial-crop"]] = self.cls2id["crop"]
            target[target == self.cls2id["partial-weed"]] = self.cls2id["weed"]
        elif self.remove_small_items:
            target[target == self.cls2id["partial-crop"]] = self.cls2id["background"]
            target[target == self.cls2id["partial-weed"]] = self.cls2id["background"]

        return input_img, target

    def get_class_number(self):
        if self.merge_small_items or self.remove_small_items:
            return 3
        else:
            return 5
        

    def visualize_data(self, num_samples=4):
        fig, axs = plt.subplots(num_samples, 3, figsize=(10, num_samples*3))

        for i in range(num_samples):
            # Select a random sample from the dataset
            idx = random.randint(0, len(self)-1)
            img, label = self[idx]

            # Convert PIL Images to numpy arrays for visualization
            img = np.array(img)
            label = np.array(label)

            # Create a color map for the label
            cmap = plt.get_cmap('tab20b', np.max(label)-np.min(label)+1)

            # Plot the image, label, and label projected onto the image
            axs[i, 0].imshow(img)
            axs[i, 0].set_title('Image')
            axs[i, 1].imshow(label, cmap=cmap, vmin=np.min(label), vmax=np.max(label))
            axs[i, 1].set_title('Label')
            axs[i, 2].imshow(img)
            axs[i, 2].imshow(label, cmap=cmap, vmin=np.min(label), vmax=np.max(label), alpha=0.5)
            axs[i, 2].set_title('Label projected onto image')

        # Remove axis labels
        for ax in axs.ravel():
            ax.axis('off')

        plt.tight_layout()
        plt.show()
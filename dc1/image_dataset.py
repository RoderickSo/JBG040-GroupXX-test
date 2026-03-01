import numpy as np
import torch
import requests
import io
from os import path
from typing import Tuple
from pathlib import Path
import os


import torch.nn.functional as F  # add this import at the top

class ImageDataset:
    """
    Creates a DataSet from numpy arrays while keeping the data
    in the more efficient numpy arrays for as long as possible and only
    converting to torchtensors when needed (torch tensors are the objects used
    to pass the data through the neural network and apply weights).
    """

    def __init__(self, x: Path, y: Path) -> None:
        # Target labels
        self.targets = ImageDataset.load_numpy_arr_from_npy(y)
        # Images
        self.imgs = ImageDataset.load_numpy_arr_from_npy(x)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        image = torch.from_numpy(self.imgs[idx] / 255).float()
        label = self.targets[idx]
        return image, label
    

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        image_np = self.imgs[idx]              # (H,W) or (1,H,W)
        label = self.targets[idx]

        image = torch.from_numpy(image_np).float() / 255.0  # scale to [0,1]

        # ensure (C,H,W)
        if image.ndim == 2:
            image = image.unsqueeze(0)  # (1,H,W)

        # resize to 224x224
        image = F.interpolate(
            image.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze(0)  # back to (1,224,224)

        return image, label

    @staticmethod
    def load_numpy_arr_from_npy(path: Path) -> np.ndarray:
        """
        Loads a numpy array from local storage.

        Input:
        path: local path of file

        Outputs:
        dataset: numpy array with input features or labels
        """

        return np.load(path)

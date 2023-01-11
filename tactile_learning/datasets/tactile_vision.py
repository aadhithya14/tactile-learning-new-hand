from copy import deepcopy
import glob
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import torchvision.transforms as T 
import torch.nn.functional as F

from torchvision.datasets.folder import default_loader as loader 
from torch.utils import data
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from tactile_learning.utils.data import load_data
from tactile_learning.utils.augmentations import crop_transform 

# Class for tactile and camera image dataset
class TactileVisionDataset(data.Dataset):
    def __init__(
        self,
        data_path,
        normalize=False,
        vision_stats=[], # Will have image means and stds
    ):
        super().__init__()
        self.roots = glob.glob(f'{data_path}/demonstration_*')
        self.roots = sorted(self.roots)

        self.data = load_data(self.roots, demos_to_use=[])

        vision_transforms = [
            T.Resize((480,640)),
            T.Lambda(crop_transform),
            T.ToTensor()
        ]
        if normalize:
            vision_transforms.append(
                T.Normalize(vision_stats[0], vision_stats[1])
            )
        self.vision_transform = T.Compose(vision_transforms)
        self.tactile_transform = T.Resize((16,16))

    def __len__(self):
        return len(self.data['tactile']['indices'])

    def _get_image(self, demo_id, image_id):
        image_root = self.roots[demo_id]
        image_path = os.path.join(image_root, 'cam_0_rgb_images/frame_{}.png'.format(str(image_id).zfill(5)))
        img = self.vision_transform(loader(image_path))
        return torch.FloatTensor(img)

    def _get_tactile_image(self, tactile_values): # shape: 15,16,3
        tactile_image = torch.FloatTensor(tactile_values) 
        tactile_image = F.pad(tactile_image, (0,0,0,0,1,0), 'constant', 0) # pads it to 16,16,3 by concatenating 0z
        tactile_image = tactile_image.view(16,4,4,3)

        tactile_image = torch.concat([
            torch.concat([tactile_image[i*4+j] for j in range(4)], dim=0)
            for i in range(4)
        ], dim=1)
        tactile_image = torch.permute(tactile_image, (2,0,1))

        return self.tactile_transform(tactile_image)

    def __getitem__(self, index):
        demo_id, tactile_id = self.data['tactile']['indices'][index]
        _, image_id = self.data['image']['indices'][index]

        # Get the tactile image
        tactile_values = self.data['tactile']['values'][demo_id][tactile_id]
        tactile_image = self._get_tactile_image(tactile_values)

        # Get the camera image
        image = self._get_image(demo_id, image_id)

        return tactile_image, image

    def getitem(self, index):
        return self.__getitem__(index) # NOTE: for debugging purposes
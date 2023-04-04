# Vision only dataset

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
from tactile_learning.utils.constants import *
from torchvision.transforms.functional import crop

class VisionDataset(data.Dataset):
    def __init__(
        self,
        data_path,
        vision_view_num
    ):
        super().__init__()
        self.roots = glob.glob(f'{data_path}/demonstration_*')
        self.roots = sorted(self.roots)
        self.data = load_data(self.roots, demos_to_use=[])
        self.vision_view_num = vision_view_num

        self.vision_transform = T.Compose([
            T.Resize((480,640)),
            T.Lambda(self._crop_transform),
            T.ToTensor(),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS),
        ])

    def __len__(self):
        return len(self.data['image']['indices'])
        
    def _get_image(self, index):
        demo_id, image_id = self.data['image']['indices'][index]
        image_root = self.roots[demo_id]
        image_path = os.path.join(image_root, 'cam_{}_rgb_images/frame_{}.png'.format(self.vision_view_num, str(image_id).zfill(5)))
        img = self.vision_transform(loader(image_path))
        return torch.FloatTensor(img)

    def __getitem__(self, index):
        vision_image = self._get_image(index)
        
        return vision_image

    def _crop_transform(self, image):
        if self.vision_view_num == 0:
            return crop(image, 0,0,480,480)
        elif self.vision_view_num == 1:
            return crop(image, 0,90,480,480)

if __name__ == '__main__':
    dset = VisionDataset(
        data_path = '/home/irmak/Workspace/Holo-Bot/extracted_data/box_handle_lifting/eval',
        vision_view_num=1
    ) 
    dataloader = data.DataLoader(dset, 
                                batch_size  = 128, 
                                shuffle     = True, 
                                num_workers = 8,
                                pin_memory  = True)

    batch = next(iter(dataloader))
    print('batch[0].shape: {}, batch[1].shape: {}'.format(
        batch[0].shape, batch[1].shape
    ))
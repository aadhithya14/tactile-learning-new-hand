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

class TactileVisionDataset(data.Dataset):
    def __init__(
        self,
        data_path,
        tactile_information_type,
        tactile_img_size,
        vision_view_num
    ):
        super().__init__()
        self.roots = glob.glob(f'{data_path}/demonstration_*')
        self.roots = sorted(self.roots)
        self.data = load_data(self.roots, demos_to_use=[])
        assert tactile_information_type in ['stacked', 'whole_hand', 'single_sensor'], 'tactile_information_type can either be "stacked", "whole_hand" or "single_sensor"'
        self.tactile_information_type = tactile_information_type
        self.vision_view_num = vision_view_num

        self.tactile_transform = T.Compose([
            T.Resize(tactile_img_size),
            T.Lambda(self._clamp_transform), # These are for normalization
            T.Lambda(self._scale_transform)
        ])
        self.vision_transform = T.Compose([
            T.Resize((480,640)),
            T.Lambda(self._crop_transform),
            T.ToTensor(),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS),
        ])

        # Set the indices for one sensor
        if tactile_information_type == 'single_sensor':
            self._preprocess_tactile_indices()
    
        # Set up the tactile image retrieval function
        if tactile_information_type == 'single_sensor':
            self._get_tactile_image = self._get_single_sensor_tactile_image
        elif tactile_information_type == 'stacked':
            self._get_tactile_image = self._get_stacked_tactile_image
        elif tactile_information_type == 'whole_hand':
            self._get_tactile_image = self._get_whole_hand_tactile_image

    def _preprocess_tactile_indices(self):
        self.tactile_mapper = np.zeros(len(self.data['tactile']['indices'])*15).astype(int)
        for data_id in range(len(self.data['tactile']['indices'])):
            for sensor_id in range(15):
                self.tactile_mapper[data_id*15+sensor_id] = data_id # Assign each finger to an index basically

    def _get_sensor_id(self, index):
        return index % 15
            
    def _get_whole_hand_tactile_image(self, tactile_values): 
        # tactile_values: (15,16,3) - turn it into 16,16,3 by concatenating 0z
        tactile_image = torch.FloatTensor(tactile_values)
        tactile_image = F.pad(tactile_image, (0,0,0,0,1,0), 'constant', 0)
        # reshape it to 4x4
        tactile_image = tactile_image.view(16,4,4,3)

        # concat for it have its proper shape
        tactile_image = torch.concat([
            torch.concat([tactile_image[i*4+j] for j in range(4)], dim=0)
            for i in range(4)
        ], dim=1)

        tactile_image = torch.permute(tactile_image, (2,0,1))
        
        return self.tactile_transform(tactile_image)
    
    def _get_stacked_tactile_image(self, tactile_values):
        tactile_image = torch.FloatTensor(tactile_values)
        tactile_image = tactile_image.view(15,4,4,3) # Just making sure that everything stays the same
        tactile_image = torch.permute(tactile_image, (0,3,1,2))
        tactile_image = tactile_image.reshape(-1,4,4) # Make 45 the channel number 
        # print('tactile_image.shape: {}'.format(tactile_image.shape)) 
        return self.tactile_transform(tactile_image)
    
    def _get_single_sensor_tactile_image(self, tactile_value):
        tactile_image = torch.FloatTensor(tactile_value) # tactile_value.shape: (16,3)
        tactile_image = tactile_image.view(4,4,3)
        tactile_image = torch.permute(tactile_image, (2,0,1))
        return self.tactile_transform(tactile_image)
    
    def __len__(self):
        if self.tactile_information_type == 'single_sensor':
            return len(self.tactile_mapper)
        else: 
            return len(self.data['tactile']['indices'])
        
    def _get_proper_tactile_value(self, index):
        if self.tactile_information_type == 'single_sensor':
            data_id = self.tactile_mapper[index]
            demo_id, tactile_id = self.data['tactile']['indices'][data_id]
            sensor_id = self._get_sensor_id(index)
            tactile_value = self.data['tactile']['values'][demo_id][tactile_id][sensor_id]
            
            return tactile_value
        
        else:
            demo_id, tactile_id = self.data['tactile']['indices'][index]
            tactile_values = self.data['tactile']['values'][demo_id][tactile_id]
            
            return tactile_values

    def _get_image(self, index):
        demo_id, image_id = self.data['image']['indices'][index]
        image_root = self.roots[demo_id]
        image_path = os.path.join(image_root, 'cam_{}_rgb_images/frame_{}.png'.format(self.vision_view_num, str(image_id).zfill(5)))
        img = self.vision_transform(loader(image_path))
        return torch.FloatTensor(img)

    def __getitem__(self, index):
        tactile_value = self._get_proper_tactile_value(index)
        tactile_image = self._get_tactile_image(tactile_value)

        vision_image = self._get_image(index)
        
        return tactile_image, vision_image

    def _scale_transform(self, image): # Transform function to map the image between 0 and 1
        image = (image - TACTILE_PLAY_DATA_CLAMP_MIN) / (TACTILE_PLAY_DATA_CLAMP_MAX - TACTILE_PLAY_DATA_CLAMP_MIN)
        return image

    def _clamp_transform(self, image):
        image = torch.clamp(image, min=TACTILE_PLAY_DATA_CLAMP_MIN, max=TACTILE_PLAY_DATA_CLAMP_MAX)
        return image

    def _crop_transform(self, image):
        if self.vision_view_num == 0:
            return crop(image, 0,0,480,480)
        elif self.vision_view_num == 1:
            return crop(image, 0,120,480,480)

if __name__ == '__main__':
    dset = TactileVisionDataset(
        data_path = '/home/irmak/Workspace/Holo-Bot/extracted_data/cup_slipping/eval',
        tactile_information_type = 'stacked',
        tactile_img_size=16,
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
    
    # cv2.imwrite('ex_image.png', )

# Class for tactile and camera image dataset
# class TactileVisionDataset(data.Dataset):
#     def __init__(
#         self,
#         data_path,
#         normalize=False,
#         tactile_only=False,
#         vision_stats=[], # Will have image means and stds
#     ):
#         super().__init__()
#         self.roots = glob.glob(f'{data_path}/demonstration_*')
#         self.roots = sorted(self.roots)

#         self.data = load_data(self.roots, demos_to_use=[])

#         vision_transforms = [
#             T.Resize((480,640)),
#             T.Lambda(crop_transform),
#             T.ToTensor()
#         ]
#         if normalize:
#             vision_transforms.append(
#                 T.Normalize(vision_stats[0], vision_stats[1])
#             )
#         self.vision_transform = T.Compose(vision_transforms)
#         self.tactile_transform = T.Resize((16,16))
#         self.tactile_only = tactile_only

#     def __len__(self):
#         return len(self.data['tactile']['indices'])

#     def _get_image(self, demo_id, image_id):
#         image_root = self.roots[demo_id]
#         image_path = os.path.join(image_root, 'cam_0_rgb_images/frame_{}.png'.format(str(image_id).zfill(5)))
#         img = self.vision_transform(loader(image_path))
#         return torch.FloatTensor(img)

#     def _get_tactile_image(self, tactile_values): # shape: 15,16,3
#         tactile_image = torch.FloatTensor(tactile_values) 
#         tactile_image = F.pad(tactile_image, (0,0,0,0,1,0), 'constant', 0) # pads it to 16,16,3 by concatenating 0z
#         tactile_image = tactile_image.view(16,4,4,3)

#         tactile_image = torch.concat([
#             torch.concat([tactile_image[i*4+j] for j in range(4)], dim=0)
#             for i in range(4)
#         ], dim=1)
#         tactile_image = torch.permute(tactile_image, (2,0,1))

#         return self.tactile_transform(tactile_image)

#     def __getitem__(self, index):
#         demo_id, tactile_id = self.data['tactile']['indices'][index]
#         _, image_id = self.data['image']['indices'][index]

#         # Get the tactile image
#         tactile_values = self.data['tactile']['values'][demo_id][tactile_id]
#         tactile_image = self._get_tactile_image(tactile_values)

#         # Get the camera image
#         if self.tactile_only:
#             return tactile_image, torch.empty(1) # Just return empty tensor if only tactile will be used

#         image = self._get_image(demo_id, image_id)
#         return tactile_image, image

#     def getitem(self, index):
#         return self.__getitem__(index) # NOTE: for debugging purposes
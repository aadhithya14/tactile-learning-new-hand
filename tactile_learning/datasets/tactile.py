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

from tactile_learning.utils.constants import *
from tactile_learning.utils.data import load_data

# from holobot.robot.allegro.allegro_kdl import AllegroKDL
from tactile_learning.datasets.preprocess import dump_video_to_images, dump_data_indices

# Class to represent tactile image
class TactileImage:
    def __init__(
        self,
        tactile_value, # (15,16,3) will be the shape
        sensor_indices = [3, 7], # Index and middle tip for now
        size = (4, 8)
    ):
        if sensor_indices is None: # It means that desired_tactile_values is equal to the tactile image itself
            desired_tactile_values = tactile_value
        else: 
            desired_tactile_values = tactile_value[sensor_indices]
        num_sensors = len(desired_tactile_values)

        # Reshape the tensor to an image according to the sensor_indices
        tactile_image = torch.FloatTensor(desired_tactile_values)
        tactile_image = tactile_image.reshape((num_sensors, size[0], int(size[1]/num_sensors), -1))
        tactile_image = torch.concat((tactile_image[0], tactile_image[1]), dim=1)
        self.tactile_image = torch.permute(tactile_image, (2,0,1))

        # Resize transform
        self.resize = T.Resize((size[1],size[1]))
        self.tactile_image = self.resize(self.tactile_image)

    def calculate_mean_std(self): # This will be used for transforms
        means, stds = [0,0,0], [0,0,0]
        for channel_num in range(self.tactile_image.shape[0]):
            means[channel_num] = self.tactile_image[channel_num,:,:].mean()
            stds[channel_num] = self.tactile_image[channel_num,:,:].std()
        return means, stds

    def plot(self):
        # Map it to 0 and 1 - not super certain this is correct
        min, max = self.tactile_image.min(), self.tactile_image.max()
        img_range = max - min
        img = (self.tactile_image - min) / img_range
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def apply_aug(self, augmentation):
        self.tactile_image = augmentation(self.tactile_image)

    def get_image(self):
        return self.tactile_image

class TactileSensorDataset(data.Dataset):
    def __init__(
        self,
        data_path,
        normalize=False,
        stats=[TACTILE_IMAGE_MEANS, TACTILE_IMAGE_STDS], # Will have image means and stds
    ):
        super().__init__()
        self.roots = glob.glob(f'{data_path}/demonstration_*')
        self.roots = sorted(self.roots)
        self.data = load_data(self.roots, demos_to_use=[])
        self.normalize = normalize
        self.normalization_transform = T.Normalize(stats[0], stats[1])
        self._preprocess_tactile_indices()

    def _preprocess_tactile_indices(self):
        self.tactile_mapper = np.zeros(len(self.data['tactile']['indices'])*15).astype(int)
        for data_id in range(len(self.data['tactile']['indices'])):
            for sensor_id in range(15):
                self.tactile_mapper[data_id*15+sensor_id] = data_id # Assign each finger to an index basically

    def _get_sensor_id(self, index):
        return index % 15

    def __len__(self):
        return len(self.tactile_mapper)

    def _get_single_tactile_image(self, tactile_value):
        tactile_image = torch.FloatTensor(tactile_value)
        tactile_image = tactile_image.view(4,4,3)
        return torch.permute(tactile_image, (2,0,1))

    def __getitem__(self, index):
        data_id = self.tactile_mapper[index]
        demo_id, tactile_id = self.data['tactile']['indices'][data_id]
        sensor_id = self._get_sensor_id(index)

        # Get the tactile image
        tactile_value = self.data['tactile']['values'][demo_id][tactile_id][sensor_id]
        tactile_image = self._get_single_tactile_image(tactile_value)

        if self.normalize:
            return self.normalization_transform(tactile_image)
        else:
            return tactile_image

    def getitem(self, index):
        return self.__getitem__(index) # NOTE: for debugging purposes

class TactileStackedDataset(data.Dataset): # Dataset that will return 16x3,4,4 images - stacked cnn
    def __init__(
        self,
        data_path,
        normalize=False,
        stats=[TACTILE_IMAGE_MEANS, TACTILE_IMAGE_STDS], # Will have image means and stds
    ):
        super().__init__()
        self.roots = glob.glob(f'{data_path}/demonstration_*')
        self.roots = sorted(self.roots)
        self.data = load_data(self.roots, demos_to_use=[])
        self.normalize = normalize
        self.normalization_transform = T.Normalize(stats[0]*15, stats[1]*15)

    def __len__(self):
        return len(self.data['tactile']['indices'])

    def _get_stacked_tactile_image(self, tactile_values):
        tactile_image = torch.FloatTensor(tactile_values)
        tactile_image = tactile_image.view(15,4,4,3) # Just making sure that everything stays the same
        tactile_image = torch.permute(tactile_image, (0,3,1,2))
        tactile_image = tactile_image.reshape(-1,4,4)

        return tactile_image

    def __getitem__(self, index):
        # Get the tactile image
        demo_id, tactile_id = self.data['tactile']['indices'][index]
        tactile_values = self.data['tactile']['values'][demo_id][tactile_id]
        tactile_image = self._get_stacked_tactile_image(tactile_values)

        if self.normalize:
            return self.normalization_transform(tactile_image)
        else:
            return tactile_image

    def getitem(self, index):
        return self.__getitem__(index) # NOTE: for debugging purposes

class TactileWholeHandDataset(data.Dataset): # Dataset that will return 16x3,4,4 images - stacked cnn
    def __init__(
        self,
        data_path,
        img_size=16,
        normalize=False,
        stats=[TACTILE_IMAGE_MEANS, TACTILE_IMAGE_STDS], # Will have image means and stds
    ):
        super().__init__()
        self.roots = glob.glob(f'{data_path}/demonstration_*')
        self.roots = sorted(self.roots)
        self.data = load_data(self.roots, demos_to_use=[])
        self.normalize = normalize
        self.resize_transform = T.Resize((img_size, img_size))
        self.normalization_transform = T.Normalize(stats[0], stats[1])

    def __len__(self):
        return len(self.data['tactile']['indices'])

    def _get_whole_hand_tactile_image(self, tactile_values): 
        # tactile_values: (15,16,3)
        # turn it into 16,16,3 by concatenating 0z
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

        return self.resize_transform(tactile_image)

    def __getitem__(self, index):
        # Get the tactile image
        demo_id, tactile_id = self.data['tactile']['indices'][index]
        tactile_values = self.data['tactile']['values'][demo_id][tactile_id]
        tactile_image = self._get_whole_hand_tactile_image(tactile_values)

        if self.normalize:
            return self.normalization_transform(tactile_image)
        else:
            return tactile_image

    def getitem(self, index):
        return self.__getitem__(index) # NOTE: for debugging purposes

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

# Minimalistick tactile dataset - it will only get fingertip of one finger
class TactileMinDataset(data.Dataset):
    # Dataset to give tactile values for two fingers and tip positions for two fingers
    def __init__(
        self,
        data_path,
        sensor_grid = (1,2), # width, height in tactile sensors
        sensor_indices = (3,7),
        allegro_finger_indices = (0,1),
        fingertip_stats = None
    ):
        super().__init__()

        # Get the demo directories 
        self.roots = glob.glob(f'{data_path}/demonstration_*')
        self.roots = sorted(self.roots)
        # print(f'roots: {self.roots}')

        # Get the indices
        self.tactile_indices = []
        self.tactile_values = []
        self.tactile_image_size = math.lcm(sensor_grid[0], sensor_grid[1]) * 4 # Image is square
        self.sensor_grid = sensor_grid
        self.sensor_indices = sensor_indices
        assert self.sensor_grid[0] == 1 and self.sensor_grid[1] == 2 # TODO: This should be fixed 
        self.allegro_finger_indices = [j for i in allegro_finger_indices for j in range(i*3,(i+1)*3)]
        self.num_sensors = self.sensor_grid[0] * self.sensor_grid[1]
        self.allegro_indices = []
        self.allegro_tip_positions = [] # Will hold 3 positions for 2 fingers (index, middle)
        self.allegro_action_indices = []
        self.allegro_actions = []

        # Load the index values 
        for root in self.roots:
            # Load the indices
            with open(os.path.join(root, 'tactile_indices.pkl'), 'rb') as f:
                self.tactile_indices += pickle.load(f)
            with open(os.path.join(root, 'allegro_indices.pkl'), 'rb') as f:
                self.allegro_indices += pickle.load(f)
            with open(os.path.join(root, 'allegro_action_indices.pkl'), 'rb') as f:
                self.allegro_action_indices += pickle.load(f)

            # Load the data
            with h5py.File(os.path.join(root, 'allegro_fingertip_states.h5'), 'r') as f:
                self.allegro_tip_positions.append(f['positions'][()][:, self.allegro_finger_indices])
            with h5py.File(os.path.join(root, 'allegro_commanded_joint_states.h5'), 'r') as f:
                self.allegro_actions.append(f['positions'][()]) # Positions are to be learned - since this is a position control
            with h5py.File(os.path.join(root, 'touch_sensor_values.h5'), 'r') as f:
                self.tactile_values.append(f['sensor_values'][()][:,sensor_indices,:,:])

        self.resize_transform = T.Resize((self.tactile_image_size, self.tactile_image_size))

        if fingertip_stats is None:
            self.fingertip_mean, self.fingertip_std = self._calculate_fingertip_mean_std()
        else:
            self.fingertip_mean, self.fingertip_std = fingertip_stats[0], fingertip_stats[1]

    def _calculate_fingertip_mean_std(self):
        num_fingers = int(len(self.allegro_finger_indices) / 3)
        all_fingertip_pos = np.zeros((len(self.allegro_indices),num_fingers,3)) # 3 for each finger - we will have positions for each axes
        for id in range(len(self.allegro_indices)):
            demo_id, allegro_id = self.allegro_indices[id]
            # Traverse through each finger
            for finger_id in range(num_fingers):
                all_fingertip_pos[id,finger_id,:] = self.allegro_tip_positions[demo_id][allegro_id][finger_id*3:(finger_id+1)*3]

        allegro_mean = all_fingertip_pos.mean(axis=(0,1))
        allegro_std = all_fingertip_pos.std(axis=(0,1))
        print('allegro_mean: {}, allegro_std: {}'.format(allegro_mean, allegro_std))

        return allegro_mean, allegro_std

    def _calculate_tactile_image_mean_std(self):
        # Will traverse through all iamges and get mean, stds per channel
        tactile_images = np.zeros((len(self.tactile_indices),
                                   3,
                                   self.tactile_image_size,
                                   self.tactile_image_size))
        for id in range(len(self.tactile_values)):
            demo_id, tactile_id = self.tactile_indices[id]
            tactile_value = self.tactile_values[demo_id][tactile_id]
            tactile_images[id,:] = self._get_tactile_image(tactile_value)

        tactile_mean = tactile_images.mean(axis=(0,2,3))
        tactile_std = tactile_images.std(axis=(0,2,3))
        print('tactile_mean: {}, tactile_std: {}'.format(tactile_mean, tactile_std))

        return tactile_mean, tactile_std

    # Method to transform tactile_value to tactile image
    def _get_tactile_image(self, tactile_value):
        tactile_image = torch.FloatTensor(tactile_value)
        tactile_image = tactile_image.reshape((
            self.sensor_grid[0] * self.sensor_grid[1],  # Total number of sensors
            4, 
            4,
            -1
        ))
        # TODO: This will only work for this grid
        tactile_image = torch.concat((tactile_image[0], tactile_image[1]), dim=1)
        tactile_image = torch.permute(tactile_image, (2,0,1))

        return self.resize_transform(tactile_image)

    def __len__(self):
        return len(self.tactile_indices)

    def __getitem__(self, index):
        demo_id, tactile_id = self.tactile_indices[index]
        _, allegro_id = self.allegro_indices[index]
        _, allegro_action_id = self.allegro_action_indices[index]

        # Get allegro values 
        allegro_tip_position = self.allegro_tip_positions[demo_id][allegro_id]
        allegro_tip_position = (allegro_tip_position - np.tile(self.fingertip_mean, 2)) / np.tile(self.fingertip_std, 2)
        allegro_action = self.allegro_actions[demo_id][allegro_action_id]

        # Get tactile image
        tactile_value = self.tactile_values[demo_id][tactile_id]
        tactile_image = self._get_tactile_image(tactile_value) # Will be returned without normalizing it - since it will be normalized

        return tactile_image, allegro_tip_position, allegro_action

    def getitem(self, index):
        return self.__getitem__(index)

class TactileLargeDataset(data.Dataset):
    # Dataset to give tactile values for two fingers and tip positions for two fingers
    def __init__(
        self,
        data_path,
        fingertip_stats = None
    ):
        super().__init__()

        # Get the demo directories 
        self.roots = glob.glob(f'{data_path}/demonstration_*')
        self.roots = sorted(self.roots)

        # Get the indices
        self.tactile_indices = []
        self.tactile_values = []
        self.tactile_image_size = 16 # This is set for now
        self.allegro_indices = []
        self.allegro_tip_positions = [] # Will hold 3 positions for 2 fingers (index, middle)
        self.allegro_action_indices = []
        self.allegro_actions = []

        # Load the index values 
        for root in self.roots:
            # Load the indices
            with open(os.path.join(root, 'tactile_indices.pkl'), 'rb') as f:
                self.tactile_indices += pickle.load(f)
            with open(os.path.join(root, 'allegro_indices.pkl'), 'rb') as f:
                self.allegro_indices += pickle.load(f)
            with open(os.path.join(root, 'allegro_action_indices.pkl'), 'rb') as f:
                self.allegro_action_indices += pickle.load(f)

            # Load the data
            with h5py.File(os.path.join(root, 'allegro_fingertip_states.h5'), 'r') as f:
                self.allegro_tip_positions.append(f['positions'][()])
            with h5py.File(os.path.join(root, 'allegro_commanded_joint_states.h5'), 'r') as f:
                self.allegro_actions.append(f['positions'][()]) # Positions are to be learned - since this is a position control
            with h5py.File(os.path.join(root, 'touch_sensor_values.h5'), 'r') as f:
                self.tactile_values.append(f['sensor_values'][()])

        self.resize_transform = T.Resize((self.tactile_image_size, self.tactile_image_size))

        if fingertip_stats is None:
            self.fingertip_mean, self.fingertip_std = self._calculate_fingertip_mean_std()
        else:
            self.fingertip_mean, self.fingertip_std = fingertip_stats[0], fingertip_stats[1]

        # self._calculate_tactile_image_mean_std() # We need to print it out for the first time

    def _calculate_fingertip_mean_std(self):
        num_fingers = 4 
        all_fingertip_pos = np.zeros((len(self.allegro_indices),num_fingers,3)) # 3 for each finger - we will have positions for each axes
        for id in range(len(self.allegro_indices)):
            demo_id, allegro_id = self.allegro_indices[id]
            # Traverse through each finger
            for finger_id in range(num_fingers):
                all_fingertip_pos[id,finger_id,:] = self.allegro_tip_positions[demo_id][allegro_id][finger_id*3:(finger_id+1)*3]

        allegro_mean = all_fingertip_pos.mean(axis=(0,1))
        allegro_std = all_fingertip_pos.std(axis=(0,1))
        print('allegro_mean: {}, allegro_std: {}'.format(allegro_mean, allegro_std))

        return allegro_mean, allegro_std

    def _calculate_tactile_image_mean_std(self):
        # Will traverse through all iamges and get mean, stds per channel
        print('calculating the mean and std')
        pbar = tqdm(total=len(self.tactile_values))
        tactile_images = np.zeros((len(self.tactile_indices),
                                   3,
                                   self.tactile_image_size,
                                   self.tactile_image_size))
        for id in range(len(self.tactile_values)):
            demo_id, tactile_id = self.tactile_indices[id]
            tactile_value = self.tactile_values[demo_id][tactile_id]
            tactile_images[id,:] = self._get_tactile_image(tactile_value)
            pbar.update(1)

        tactile_mean = tactile_images.mean(axis=(0,2,3))
        tactile_std = tactile_images.std(axis=(0,2,3))
        pbar.close()
        print('tactile_mean: {}, tactile_std: {}'.format(tactile_mean, tactile_std))
        return tactile_mean, tactile_std

    # Method to transform tactile_value to tactile image
    def _get_tactile_image(self, tactile_values): 
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

    def __len__(self):
        return len(self.tactile_indices)

    def __getitem__(self, index):
        demo_id, tactile_id = self.tactile_indices[index]
        _, allegro_id = self.allegro_indices[index]
        _, allegro_action_id = self.allegro_action_indices[index]

        # Get allegro values 
        allegro_tip_position = self.allegro_tip_positions[demo_id][allegro_id]
        allegro_tip_position = (allegro_tip_position - np.tile(self.fingertip_mean, 4)) / np.tile(self.fingertip_std, 4)
        allegro_action = self.allegro_actions[demo_id][allegro_action_id]

        # Get tactile image
        tactile_value = self.tactile_values[demo_id][tactile_id]
        tactile_image = self._get_tactile_image(tactile_value) # Will be returned without normalizing it - since it will be normalized

        return tactile_image, allegro_tip_position, allegro_action

    def getitem(self, index):
        return self.__getitem__(index)



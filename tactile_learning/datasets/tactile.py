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

from torchvision.datasets.folder import default_loader as loader 
from torch.utils import data
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

# from holobot.robot.allegro.allegro_kdl import AllegroKDL
from tactile_learning.datasets.preprocess import dump_video_to_images, dump_data_indices


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

# Class to traverse through the data saved and get the data according to the timestamps
# all of them should be saved with
# TODO: Separate every data type - image, allegro, kinova, tactile 
class TactileFullDataset(data.Dataset):
    def __init__(self,
        data_path,
        tactile_stats=None, # Tuple of mean and std of the tactile sensor information
        allegro_stats=None # Touple of mean and std of the allegro hand joint
    ):

        # Get the demonstration directories
        self.roots = glob.glob(f'{data_path}/demonstration_*') # TODO: change this in the future
        self.roots = sorted(self.roots)

        print('roots: {}'.format(self.roots))

        # Get the dumped indices and the positions
        self.tactile_indices = []
        self.tactile_values = [] 
        self.image_indices = [] 
        self.allegro_indices = []
        self.allegro_positions = []
        self.allegro_actions = []
        for demo_id, root in enumerate(self.roots):
            # Load the indices
            with open(os.path.join(root, 'tactile_indices.pkl'), 'rb') as f:
                self.tactile_indices += pickle.load(f)
            with open(os.path.join(root, 'image_indices.pkl'), 'rb') as f:
                self.image_indices += pickle.load(f)
            with open(os.path.join(root, 'allegro_indices.pkl'), 'rb') as f:
                self.allegro_indices += pickle.load(f)

            # Load the data
            with h5py.File(os.path.join(root, 'allegro_joint_states.h5'), 'r') as f:
                self.allegro_positions.append(f['positions'][()])
            with h5py.File(os.path.join(root, 'allegro_commanded_joint_states.h5'), 'r') as f:
                print(
                    'allegro commanded joint state keys: {}'.format(
                        f.keys()
                    )
                )
                self.allegro_actions.append(f['positions'][()]) # Positions are to be learned - since this is a position control
            with h5py.File(os.path.join(root, 'touch_sensor_values.h5'), 'r') as f:
                self.tactile_values.append(f['sensor_values'][()])

        self.transform = T.Compose([
                T.Resize((480,640)),
                T.CenterCrop((480,480)), # TODO: Burda 480,480 yap bunu
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        # Get the mean and stds of the 
        if tactile_stats is None:
            self._tactile_mean, self._tactile_std = self._calculate_tactile_mean_std()
            # Dump the tactile mean to the given data_path if it doesn't exist 
            with open(os.path.join(data_path, 'tactile_stats.pkl'), 'wb') as f:
                tactile_stats = np.stack((self._tactile_mean, self._tactile_std), axis=0)
                pickle.dump(tactile_stats, f)
        else:
            self._tactile_mean, self._tactile_std = tactile_stats
            
        if allegro_stats is None:
            self._allegro_mean, self._allegro_std = self._calculate_allegro_mean_std()
            # Dump the allegro mean and std to the given data path if it doesn't exist 
            with open(os.path.join(data_path, 'allegro_stats.pkl'), 'wb') as f:
                allegro_stats = np.stack((self._allegro_mean, self._allegro_std), axis=0)
                pickle.dump(allegro_stats, f)
        else:
            self._allegro_mean, self._allegro_std = allegro_stats

    def __len__(self):
        return len(self.tactile_indices)

    def _get_image(self, demo_id, image_id):
        image_root = self.roots[demo_id]
        image_path = os.path.join(image_root, 'cam_0_rgb_images/frame_{}.png'.format(str(image_id).zfill(5)))

        img = self.transform(loader(image_path))
        return torch.FloatTensor(img)

    def getitem(self, id):
        return self.__getitem__(id)

    def __getitem__(self, id):
        demo_id, tactile_id = self.tactile_indices[id]
        _, allegro_id = self.allegro_indices[id]
        _, image_id = self.image_indices[id]
        
        # Get the tactile information
        tactile_info = self.tactile_values[demo_id][tactile_id]
        tactile_info = (tactile_info - self._tactile_mean) / self._tactile_std

        # Get the joint positions
        allegro_pos = self.allegro_positions[demo_id][allegro_id]
        allegro_pos = (allegro_pos - self._allegro_mean) / self._allegro_std

        # Get the actions 
        actions = self.allegro_actions[demo_id][allegro_id]
        # TODO: Normalize it?

        # Get the image 
        image = self._get_image(demo_id, image_id)

        return image, tactile_info, allegro_pos, actions

    def _calculate_tactile_mean_std(self):
        all_tactile_info = np.zeros((len(self.tactile_indices), 15,16,3))
        for id in range(len(self.tactile_indices)):
            demo_id, tactile_id = self.tactile_indices[id]
            all_tactile_info[id] = self.tactile_values[demo_id][tactile_id]

        tactile_mean = all_tactile_info.mean(axis=0)
        tactile_std = all_tactile_info.std(axis=0)
        return tactile_mean, tactile_std

    def _calculate_allegro_mean_std(self):
        all_joint_pos = np.zeros((len(self.allegro_indices), 16))
        for id in range(len(self.allegro_indices)):
            demo_id, allegro_id = self.allegro_indices[id]
            all_joint_pos[id] = self.allegro_positions[demo_id][allegro_id]

        allegro_mean = all_joint_pos.mean(axis=0)
        allegro_std = all_joint_pos.std(axis=0)
        return allegro_mean, allegro_std

if __name__ == '__main__':
    dset = TactileMinDataset(
        data_path = '/home/irmak/Workspace/Holo-Bot/extracted_data/joystick'
    )
    dset._calculate_fingertip_mean_std()
    dset._calculate_tactile_image_mean_std()
    # data_loader = data.DataLoader(dset, batch_size=64)
    # batch = next(iter(data_loader))
    # # batch = dset.getitem(0)
    # tactile_img = batch[0]
    # print(tactile_img.shape)
    # tactile_img, tip_pos, action = dset.getitem(0)
    # print()
    # dset = TactileFullDataset(data_path='/home/irmak/Workspace/Holo-Bot/extracted_data/logitech_mouse')
    # image, tactile_info, robot_state, actions = dset.getitem(0)
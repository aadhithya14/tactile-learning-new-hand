from copy import deepcopy
import glob
import h5py
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

# Class to traverse through the data saved and get the data according to the timestamps
# all of them should be saved with
# TODO: Separate every data type - image, allegro, kinova, tactile 
class TactileDataset(data.Dataset):
    def __init__(self,
        data_path,
        tactile_stats=None, # Tuple of mean and std of the tactile sensor information
        allegro_stats=None # Touple of mean and std of the allegro hand joint
    ):

        # Get the demonstration directories
        # self.roots = glob.glob(f'{data_path}/demonstration_*') # TODO: change this in the future
        # self.roots = sorted(self.roots)

        # print('roots: {}'.format(self.roots))

        self.roots = ['/home/irmak/Workspace/Holo-Bot/extracted_data/demonstration_17',
                      '/home/irmak/Workspace/Holo-Bot/extracted_data/demonstration_18']
        self.roots = sorted(self.roots)

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
                self.allegro_actions.append(f['velocitys'][()])
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
    dset = TactileDataset(data_path='/home/irmak/Workspace/Holo-Bot/extracted_data')
    image, tactile_info, robot_state, actions = dset.getitem(0)
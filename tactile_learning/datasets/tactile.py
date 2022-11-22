from copy import deepcopy
import glob
import h5py
import numpy as np
import os
import pickle
import torch
import torchvision.transforms as T 
from torchvision.datasets.folder import default_loader as loader 

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from holobot.robot.allegro.allegro_kdl import AllegroKDL
from tactile_learning.datasets.preprocess import dump_video_to_images

# Class to traverse through the data saved and get the data according to the timestamps
# all of them should be saved with 
class TactileDataset():
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
        self._allegro_kdl_solver = AllegroKDL()
        
        # Load the tactile and allegro state data for all the roots
        self.allegro_data = [] 
        self.tactile_data = []
        self.image_metadata = []
        self._load_state_data()

        # Get the indexing 
        self._data_indexing = { # Dictionary will hold the indices for each general index in _get_item 
            'tactile_indices': [], # each element will be (demo_id, frame_num)
            'image_indices': [],
            'robot_state_indices': []
        }
        self._get_dataset_info()

        self.transform = T.Compose([
                T.Resize((480,640)),
                T.CenterCrop((480,480)), # TODO: Burda 480,480 yap bunu
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        # Get the mean and stds of the 
        if tactile_stats is None:
            self._tactile_mean, self._tactile_std = self._calculate_tactile_mean_std()
        else:
            self._tactile_mean, self._tactile_std = tactile_stats
            
        if allegro_stats is None:
            self._allegro_mean, self._allegro_std = self._calculate_allegro_mean_std()
        else:
            self._allegro_mean, self._allegro_std = allegro_stats


    def __len__(self):
        return len(self._data_indexing['tactile_indices'])

    def _get_image(self, demo_id, image_id):
        image_root = self.roots[demo_id]
        image_path = os.path.join(image_root, 'cam_0_rgb_images/frame_{}.png'.format(str(image_id).zfill(5)))
        # print('image_path: {}'.format(image_path))

        img = self.transform(loader(image_path))
        return torch.FloatTensor(img)

    def getitem(self, id):
        return self.__getitem__(id)

    def __getitem__(self, id):
        tac_demo_id, tactile_id = self._data_indexing['tactile_indices'][id]
        rs_demo_id, robot_state_id = self._data_indexing['robot_state_indices'][id]
        img_demo_id, image_id = self._data_indexing['image_indices'][id]

        # Get the tactile information
        tactile_info = self.tactile_data[tac_demo_id]['sensor_values'][tactile_id]
        tactile_info = (tactile_info - self._tactile_mean) / self._tactile_std

        # Get the joint positions
        robot_state = self.allegro_data[rs_demo_id]['positions'][robot_state_id]
        robot_state = (robot_state - self._allegro_mean) / self._allegro_std

        # Get the actions 
        actions = self.allegro_data[rs_demo_id]['actions'][robot_state_id]
        # TODO: Normalize it?

        # Get the image 
        image = self._get_image(img_demo_id, image_id)

        return image, tactile_info, robot_state, actions

    def _calculate_tactile_mean_std(self):
        all_tactile_info = np.zeros((len(self._data_indexing['tactile_indices']), 15,16,3))
        for id in range(len(self._data_indexing['tactile_indices'])):
            demo_id, tactile_id = self._data_indexing['tactile_indices'][id]
            all_tactile_info[id] = self.tactile_data[demo_id]['sensor_values'][tactile_id]

        tactile_mean = all_tactile_info.mean(axis=0)
        tactile_std = all_tactile_info.std(axis=0)

        # print('tactile_mean.shape: {}, tactile_std.shape: {}'.format(
        #     tactile_mean.shape, tactile_std.shape
        # ))
        # print('tactile_mean: {}, tactile_std: {}'.format(tactile_mean, tactile_std))

        return tactile_mean, tactile_std
        # pass 

    def _calculate_allegro_mean_std(self):
        all_joint_pos = np.zeros((len(self._data_indexing['robot_state_indices']), 16))
        for id in range(len(self._data_indexing['robot_state_indices'])):
            demo_id, allegro_id = self._data_indexing['robot_state_indices'][id]
            all_joint_pos[id] = self.allegro_data[demo_id]['positions'][allegro_id]

        allegro_mean = all_joint_pos.mean(axis=0)
        allegro_std = all_joint_pos.std(axis=0)

        # print('allegro_mean.shape: {}, allegro_std.shape: {}'.format(
        #     allegro_mean.shape, allegro_std.shape
        # ))
        # print('allegro_mean: {}, allegro_std: {}'.format(allegro_mean, allegro_std))

        return allegro_mean, allegro_std

    # Load the allegro and tactile data to memory
    def _load_state_data(self):
        
        for root in self.roots:
            tactile_path = os.path.join(root, 'touch_sensor_values.h5')
            allegro_states_path = os.path.join(root, 'allegro_joint_states.h5')
            allegro_commands_path = os.path.join(root, 'allegro_commanded_joint_states.h5')  
            image_metadata_path = os.path.join(root, 'cam_0_rgb_video.metadata')

            # Add the state
            with h5py.File(allegro_states_path, 'r') as f:
                self.allegro_data.append({
                    'timestamps': f['timestamps'][()],
                    'positions': f['positions'][()]
                })

            # Add the commands 
            with h5py.File(allegro_commands_path, 'r') as f:
                # print('f.keys(): {} in {}'.format(
                #     f.keys(), allegro_commands_path
                # ))
                self.allegro_data[-1]['actions'] = f['velocitys'][()]
            
            with h5py.File(tactile_path, 'r') as f:
                self.tactile_data.append({
                    'timestamps': f['timestamps'][()],
                    'sensor_values': f['sensor_values'][()]
                })
            with open(image_metadata_path, 'rb') as f:
                image_metadata = pickle.load(f)
                self.image_metadata.append({
                    'timestamps': self._traverse_image_timestamps(image_metadata['timestamps']) 
                })
                
    # Method to traverse the number of frames in each demonstration
    def _get_dataset_info(self):
        # self.
        # Dataset will first get the first tactile sensor since they start 3 seconds later
        # And then find the timestamps where there was a significat difference between allegro readings 
        for demo_id, root in enumerate(self.roots):
            # For each demo, find the earliest timestamp for tactile sensor
            allegro_id = 0 # These ids are used to keep track when we're finding the closest timestamps
            image_id = 0 
            tactile_id = 0 

            # Get the earliest timestamp for tactile sensor and find curresponding allegro pos ids for that
            tactile_timestamp = self.tactile_data[demo_id]['timestamps'][0]
            allegro_pos_id = self._get_closest_id(allegro_id, tactile_timestamp, self.allegro_data[demo_id]['timestamps'])
            image_id = self._get_closest_id(image_id, tactile_timestamp, self.image_metadata[demo_id]['timestamps'])

            # Save the ids
            self._data_indexing['tactile_indices'].append((demo_id, tactile_id))
            self._data_indexing['robot_state_indices'].append((demo_id, allegro_pos_id))
            self._data_indexing['image_indices'].append((demo_id, image_id))

            while(True):
                # Find the next allegro pos id with a change
                allegro_pos_id = self._find_next_allegro_id(demo_id, allegro_pos_id)
                if allegro_pos_id is None:
                    break
                
                # Find the closest timestamps with the given allegro pos id
                allegro_timestamp = self.allegro_data[demo_id]['timestamps'][allegro_pos_id]
                tactile_id = self._get_closest_id(tactile_id, allegro_timestamp, self.tactile_data[demo_id]['timestamps'])
                image_id = self._get_closest_id(image_id, allegro_timestamp, self.image_metadata[demo_id]['timestamps'])

                self._data_indexing['tactile_indices'].append((demo_id, tactile_id))
                self._data_indexing['robot_state_indices'].append((demo_id, allegro_pos_id))
                self._data_indexing['image_indices'].append((demo_id, image_id)) 

                if image_id == len(self.image_metadata[demo_id]['timestamps']) or \
                   tactile_id == len(self.tactile_data[demo_id]['timestamps']) or \
                   allegro_id == len(self.allegro_data[demo_id]['timestamps']):

                    return

    def _get_closest_id(self, curr_id, desired_timestamp, all_timestamps):
        # Find the closest timestamp to desired timetamp in all_timestamps - starting from curr_id
        for i in range(curr_id, len(all_timestamps)):
            if all_timestamps[i] > desired_timestamp:
                return i # Find the first timestamp that is after that

    # Traverse through the allegro data in the given demo and position and find the next hand reading
    # with a significant change
    def _find_next_allegro_id(self, demo_id, pos_id):
        # demo_id, pos_id: since the data is saved as demo, pos - these two indices give the indices of the current allegro state
        # returns the timestamp of the next good allegro hand pose
        old_allegro_pos = self.allegro_data[demo_id]['positions'][pos_id]
        old_allegro_fingertip_pos = self._get_fingertip_coords(old_allegro_pos)
        for i in range(pos_id, len(self.allegro_data[demo_id]['positions'])):
            curr_allegro_fingertip_pos = self._get_fingertip_coords(self.allegro_data[demo_id]['positions'][i])
            step_size = np.linalg.norm(old_allegro_fingertip_pos - curr_allegro_fingertip_pos)
            if step_size > 0.02: 
                return i

    # Realsense timestamps are given not in seconds + nano seconds
    def _traverse_image_timestamps(self, timestamps):
        for i in range(len(timestamps)):
            timestamps[i] /= 1000
        return timestamps

    # Method to return the fingertip coordinates given joint angle positions
    def _get_fingertip_coords(self, joint_positions): # - NOTE: Taken from the allegro library
        index_coords = self._allegro_kdl_solver.finger_forward_kinematics('index', joint_positions[:4])[0]
        middle_coords = self._allegro_kdl_solver.finger_forward_kinematics('middle', joint_positions[4:8])[0]
        ring_coords = self._allegro_kdl_solver.finger_forward_kinematics('ring', joint_positions[8:12])[0]
        thumb_coords = self._allegro_kdl_solver.finger_forward_kinematics('thumb', joint_positions[12:16])[0]

        finger_tip_coords = np.hstack([index_coords, middle_coords, ring_coords, thumb_coords])
        return np.array(finger_tip_coords)

if __name__ == '__main__':
    dset = TactileDataset(data_path='/home/irmak/Workspace/Holo-Bot/extracted_data')
    # image, tactile_info, robot_state, actions = dset.getitem(0)

    # image, tactile_info, robot_state = dset.getitem(1)

    # Dump the videos to frames
    # dump_video_to_images(root='/home/irmak/Workspace/Holo-Bot/extracted_data/demonstration_17')
from copy import deepcopy
import glob
import h5py
import numpy as np
import os
import pickle
# import random
import torch
# import torch.utils.data as Dataset
import torchvision.transforms as T 

from omegaconf import DictConfig, OmegaConf

from holobot.robot.allegro.allegro_kdl import AllegroKDL
from tactile_learning.datasets.preprocess import dump_video_to_images

# Class to traverse through the data saved and get the data according to the timestamps
# all of them should be saved with 
class TactileDataset():
    def __init__(self, data_path):

        # Get the demonstration directories
        # self.roots = glob.glob(f'{data_path}/demonstration_*') # TODO: change this in the future
        # self.roots = sorted(self.roots)

        # print('roots: {}'.format(self.roots))

        self.roots = ['/home/irmak/Workspace/Holo-Bot/extracted_data/demonstration_17',
                      '/home/irmak/Workspace/Holo-Bot/extracted_data/demonstration_18']
        self._allegro_kdl_solver = AllegroKDL()
        
        # Load the tactile and allegro state data for all the roots
        self._load_state_data()
        # print('allegro_data timestamp len: {}, tactile timestamps len: {}, image timestamps len: {}'.format(
        #     len(self.allegro_data[0]['timestamps']), len(self.tactile_data[0]['timestamps']), len(self.image_metadata[0]['timestamps'])
        # ))

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

    def _getitem_(self, id):
        tac_demo_id, tactile_id = self._data_indexing['tactile_indices'][id]
        rs_demo_id, robot_state_id = self._data_indexing['robot_state_indices'][id]
        img_demo_id, image_id = self._data_indexing['image_indices'][id]

        


    # Load the allegro and tactile data to memory
    def _load_state_data(self):
        self.allegro_data = [] 
        self.tactile_data = []
        self.image_metadata = []
        for root in self.roots:
            tactile_path = os.path.join(root, 'touch_sensor_values.h5')
            allegro_states_path = os.path.join(root, 'allegro_joint_states.h5')  
            image_metadata_path = os.path.join(root, 'cam_0_rgb_video.metadata')

            with h5py.File(allegro_states_path, 'r') as f:
                self.allegro_data.append({
                    'timestamps': f['timestamps'][()],
                    'positions': f['positions'][()]
                })
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

            # print('tactile_timestamp: {}, allegro_timestamp[{}]: {}, image_timestamp[{}]: {}'.format(
            #     tactile_timestamp, allegro_pos_id, self.allegro_data[demo_id]['timestamps'][allegro_pos_id],
            #     image_id, self.image_metadata[demo_id]['timestamps'][image_id]
            # ))
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

                # print('tactile_timestamp[{}]: {}, allegro_timestamp[{}]: {}, image_timestamp[{}]: {}'.format(
                #     tactile_id, self.tactile_data[demo_id]['timestamps'][tactile_id],
                #     allegro_pos_id, self.allegro_data[demo_id]['timestamps'][allegro_pos_id],
                #     image_id, self.image_metadata[demo_id]['timestamps'][image_id]
                # ))

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

    # Dump the videos to frames
    # dump_video_to_images(root='/home/irmak/Workspace/Holo-Bot/extracted_data/demonstration_17')
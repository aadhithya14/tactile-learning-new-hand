import h5py
import numpy as np
import os
import pickle
import torch

from tqdm import tqdm

# Method to load all the data from given roots and return arrays for it
def load_data(roots, demos_to_use=[]):
    roots = sorted(roots)

    tactile_indices = [] 
    allegro_indices = []
    allegro_action_indices = [] 
    kinova_indices = []
    image_indices = []
    
    tactile_values = {}
    allegro_tip_positions = {} 
    allegro_joint_positions = {}
    allegro_actions = {}
    kinova_states = {}

    for demo_id,root in enumerate(roots): 
        demo_num = int(root.split('/')[-1].split('_')[-1])
        # print(f'len(demos_to_use): {len(demos_to_use)}')
        if (len(demos_to_use) > 0 and demo_num in demos_to_use) or (len(demos_to_use) == 0): # If it's empty then it will be ignored
            # print(f'USING DEMO: {demo_num} [DEMO_ID: {demo_id} - IN ROOT: {root}')
            with open(os.path.join(root, 'tactile_indices.pkl'), 'rb') as f:
                tactile_indices += pickle.load(f)
            with open(os.path.join(root, 'allegro_indices.pkl'), 'rb') as f:
                allegro_indices += pickle.load(f)
            with open(os.path.join(root, 'allegro_action_indices.pkl'), 'rb') as f:
                allegro_action_indices += pickle.load(f)
            with open(os.path.join(root, 'kinova_indices.pkl'), 'rb') as f:
                kinova_indices += pickle.load(f)
            with open(os.path.join(root, 'image_indices.pkl'), 'rb') as f:
                image_indices += pickle.load(f)

            # Load the data
            with h5py.File(os.path.join(root, 'allegro_fingertip_states.h5'), 'r') as f:
                allegro_tip_positions[demo_id] = f['positions'][()]
            with h5py.File(os.path.join(root, 'allegro_joint_states.h5'), 'r') as f:
                allegro_joint_positions[demo_id] = f['positions'][()]
            with h5py.File(os.path.join(root, 'allegro_commanded_joint_states.h5'), 'r') as f:
                allegro_actions[demo_id] = f['positions'][()] # Positions are to be learned - since this is a position control
            with h5py.File(os.path.join(root, 'touch_sensor_values.h5'), 'r') as f:
                tactile_values[demo_id] = f['sensor_values'][()]
            with h5py.File(os.path.join(root, 'kinova_cartesian_states.h5'), 'r') as f:
                state = np.concatenate([f['positions'][()], f['orientations'][()]], axis=1)     
                kinova_states[demo_id] = state

    data = dict(
        tactile = dict(
            indices = tactile_indices,
            values = tactile_values
        ),
        allegro_joint_states = dict(
            indices = allegro_indices, 
            values = allegro_joint_positions
        ),
        allegro_tip_states = dict(
            indices = allegro_indices, 
            values = allegro_tip_positions
        ),
        allegro_actions = dict(
            indices = allegro_action_indices,
            values = allegro_actions
        ),
        kinova = dict( 
            indices = kinova_indices, 
            values = kinova_states
        ), 
        image = dict( 
            indices = image_indices
        )
    )

    return data


def get_image_stats(len_image_dataset, image_loader):
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for inputs in tqdm(image_loader):
        psum    += inputs.sum(axis = [0, 2, 3])
        psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])

    # pixel count
    count = len_image_dataset * 480 * 480

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))
        
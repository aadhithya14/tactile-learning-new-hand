# Script to deploy already created demo
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image as im
from omegaconf import OmegaConf
from tqdm import tqdm 
from torchvision.datasets.folder import default_loader as loader

from holobot.constants import *
from holobot.robot.allegro.allegro_kdl import AllegroKDL

from tactile_learning.deployment.load_models import load_model
from tactile_learning.deployment.nn_buffer import NearestNeighborBuffer
from tactile_learning.models.knn import KNearestNeighbors, ScaledKNearestNeighbors
from tactile_learning.utils.augmentations import crop_transform
from tactile_learning.utils.constants import *
from tactile_learning.utils.data import load_data
from tactile_learning.utils.tactile_image import get_tactile_image
from tactile_learning.utils.visualization import *

class RunDemo:
    def __init__(
        self,
        data_path, # root in string
        demo_to_run,
        apply_allegro_states = False, # boolean to indicate if we should apply commanded allegro states or actual allegro states
        robots = ['allegro', 'kinova'],
    ):

        roots = glob.glob(f'{data_path}/demonstration_*')
        roots = sorted(roots)
        self.data = load_data(roots, demos_to_use=[demo_to_run])
        self.state_id = 0
        self.allegro_action_key = 'allegro_joint_states' if apply_allegro_states else 'allegro_actions'
        self.robots = robots
        # print('self.data[allegro_states][values][0][:]: {}'.format(self.data['allegro_states']['values'][0][:]))
        # print('self.allegro_action_key: {}'.format(self.allegro_action_key))
        print('state indices: {}'.format(self.data['allegro_joint_states']['indices']))
        print('action indices: {}'.format(self.data['allegro_actions']['indices']))

    def get_action(self, tactile_values, recv_robot_state, visualize=False):
        demo_id, action_id = self.data[self.allegro_action_key]['indices'][self.state_id] 
        allegro_action = self.data[self.allegro_action_key]['values'][demo_id][action_id] # Get the next commanded action (commanded actions are saved in that timestamp)
        action = dict(
            allegro = allegro_action
        )

        if 'kinova' in self.robots:
            _, kinova_id = self.data['kinova']['indices'][self.state_id] 
            kinova_action = self.data['kinova']['values'][demo_id][kinova_id] # Get the next saved kinova_state
            action['kinova'] = kinova_action

        print(f'applying the {self.state_id}th demo action: {action}')

        self.state_id += 1

        return action

    def save_deployment(self): # We don't really need to do anything here
        pass
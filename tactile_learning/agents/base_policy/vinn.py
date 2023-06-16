import datetime
import glob
import os
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from pathlib import Path
from PIL import Image

from torchvision.utils import save_image
from torchvision import transforms as T
# from agent.encoder import Encoder

from tactile_learning.models import *
from tactile_learning.utils import *
from tactile_learning.tactile_data import *
from tactile_learning.deployers.utils.nn_buffer import NearestNeighborBuffer

from holobot.robot.allegro.allegro_kdl import AllegroKDL

from .base_policy import BasePolicy

# Will choose the first frame demonstration and move from there
class VINN(BasePolicy):
    def __init__(
        self,
        expert_demos,
        tactile_repr_size,
        image_out_dir,
        device,
        max_steps,
        image_model_type='byol',
        **kwargs
    ):
        
        print('kwargs in VINN Basepolicy: {}, expert_demos: {}'.format(expert_demos, kwargs ))

        self.device = device
        self.expert_id = 0
        self.set_expert_demos(expert_demos)

        # Load the image encoders
        _, self.image_encoder, _ = init_encoder_info(
            device = self.device,
            out_dir = image_out_dir,
            encoder_type = 'image',
            model_type = image_model_type
        )

        # Set the image normalization
        self.image_normalize = T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)
        
        # Get all the representations
        self._get_all_representations()
        self.max_steps = max_steps

        # Test turn id method because it's very dependant
        print('### TESTING TURN ID METHOD ON ALL REPRESENTATIONS ###')
        for i in range(len(self.all_representations)):
            exp_id, frame_id = self._turn_id(
                frame_id=i, frame2expert=True)
            tested_overall_id = self._turn_id(
                frame_id = frame_id,
                expert_id = exp_id,
                expert2frame=True)
            print('ground truth frame id: {}, exp_id: {}, frame_id: {}, turned_overall_frame_id: {}'.format(
                i, exp_id, frame_id, tested_overall_id
            ))

            assert i == tested_overall_id, f'Ground truth frame id: {i} doesnt match test frame id: {tested_overall_id}'

        self.nn_k = 10 # We set these to what works for now - it never becomes more than 10 in our tasks
        self.buffer = NearestNeighborBuffer(10)
        self.knn = ScaledKNearestNeighbors(
            self.all_representations, # Both the input and the output of the nearest neighbors are
            self.all_representations,
            ['image', 'tactile'],
            [1, 1], # Could have tactile doubled
            tactile_repr_size=tactile_repr_size
        )

    def _get_all_representations(self):
        print('Getting all representations')
        all_representations = []

        all_reprs_sum = sum([len(self.expert_demos[i]['image_obs']) for i in range(len(self.expert_demos))])
        pbar = tqdm(total=all_reprs_sum)
        for expert_id in range(len(self.expert_demos)):
            # pbar = tqdm(total=len(self.expert_demos[expert_id]['image_obs']))
            image_reprs = self.image_encoder(self.expert_demos[expert_id]['image_obs'].to(self.device))
            tactile_reprs = self.expert_demos[expert_id]['tactile_repr'].to(self.device)
            expert_reprs = torch.concat([image_reprs, tactile_reprs], dim=-1).detach().cpu()
            
            all_representations.append(expert_reprs)
            pbar.update(len(self.expert_demos[expert_id]['image_obs']))

        pbar.close()
        self.all_representations = torch.concat(all_representations, dim=0) # Here, since each representation 
        print('all_representations.shape: {}'.format(self.all_representations.shape))

        # pbar = tqdm(total=len(self.data['tactile']['indices']))
        # for index in range(len(self.data['tactile']['indices'])):
        #     # Get the representation data
        #     demo_id, tactile_id = self.data['tactile']['indices'][index]
        #     tactile_value = self.data['tactile']['values'][demo_id][tactile_id]
        #     tactile_repr = self.tactile_repr.get(tactile_value, detach=False)

        #     _, image_id = self.data['image']['indices'][index]
        #     image = load_dataset_image(
        #         data_path = self.data_path, 
        #         demo_id = demo_id, 
        #         image_id = image_id,
        #         view_num = self.view_num,
        #         transform = self.image_transform
        #     )
        #     image_repr = self.vinn_encoder(image.unsqueeze(0).to(self.device)).squeeze()
        #     all_repr = torch.concat([image_repr, tactile_repr], dim=-1).detach().cpu()
        #     all_representations.append(all_repr)

        #     pbar.update(1)

        # pbar.close()
        # self.all_representations = torch.stack(all_representations, dim=0)
        
    # def act(self, obs, episode_step, **kwargs):
    #     # Get the expert id
    #     self.expert_id, _ = self._get_closest_expert_id(self, obs)

    #     # Use expert_demos for base action retrieval
    #     is_done = False
    #     if episode_step >= len(self.expert_demos[self.expert_id]['actions']):
    #         episode_step = len(self.expert_demos[self.expert_id]['actions'])-1
    #         is_done = True

    #     action = self.expert_demos[self.expert_id]['actions']

    #     return action, is_done
    
    def _turn_id(self, frame_id, expert_id=None, expert2frame=False, frame2expert=False):
        if expert2frame: # It's assumed that the given id is frame based
            assert not expert_id is None, 'Expert ID cannot be empty if expert2frame is true'
            overall_frame_id = 0
            for curr_exp_id in range(len(self.expert_demos)):
                if curr_exp_id == expert_id:
                    overall_frame_id += frame_id
                    return overall_frame_id
                
                overall_frame_id += len(self.expert_demos[curr_exp_id]['image_obs'])

        elif frame2expert:
            # Assumed that all of the demos are inputted
            overall_frame_id = frame_id
            for curr_exp_id in range(len(self.expert_demos)):
                if overall_frame_id < len(self.expert_demos[curr_exp_id]['image_obs']):
                    return curr_exp_id, overall_frame_id
                
                overall_frame_id -= len(self.expert_demos[curr_exp_id]['image_obs'])

        return None # Force the model to return error if proper parameters are not inputted

    def act(self, obs, episode_step, **kwargs):
        # Get the current representation
        image_obs = obs['image_obs'].unsqueeze(0) / 255.
        tactile_repr = obs['tactile_repr'].numpy()
        image_obs = self.image_normalize(image_obs.float()).to(self.device)
        image_repr = self.image_encoder(image_obs).detach().cpu().numpy().squeeze()
        curr_repr = np.concatenate([image_repr, tactile_repr], axis=0)

        # Choose the action with the buffer 
        _, nn_idxs, _ = self.knn.get_k_nearest_neighbors(curr_repr, k=self.nn_k)
        id_of_nn = self.buffer.choose(nn_idxs)
        nn_id = nn_idxs[id_of_nn]
        # if nn_id+1 >= len(self.data['allegro_actions']['indices']): # If the chosen action is the action after the last action
        #     nn_idxs = np.delete(nn_idxs, id_of_nn)
        #     id_of_nn = self.buffer.choose(nn_idxs)
        #     nn_id = nn_idxs[id_of_nn]

        # Check if the closest neighbor is the last frame
        is_done = False
        curr_demo_id, _ = self._turn_id(frame_id=nn_id, frame2expert=True)
        next_demo_id, next_frame_id = self._turn_id(frame_id=nn_id+1, frame2expert=True)
        if next_demo_id != curr_demo_id:
            is_done = True
            nn_id -= 1 # Just send the last frame action
        elif episode_step > self.max_steps: 
            is_done = True

        action = self.expert_demos[next_demo_id]['actions'][next_frame_id]
        # demo_id, action_id = self.data['allegro_actions']['indices'][nn_id+1]  # Get the next commanded action (commanded actions are saved in that timestamp)
        # nn_allegro_action = self.data['allegro_actions']['values'][demo_id][action_id]
        # _, kinova_id = self.data['kinova']['indices'][nn_id+1] # Get the next kinova state (which is for kinova robot the same as the next commanded action)
        # nn_kinova_action = self.data['kinova']['values'][demo_id][kinova_id]

        print('EPISODE STEP: {} self.max_vinn_steps: {} ID OF NN: {} NEAREST NEIGHBOR DEMO ID: {}, IS DONE IN VINN: {}'.format(
            episode_step, self.max_steps, id_of_nn, next_demo_id, is_done))

        # action = np.concatenate([nn_allegro_action, nn_kinova_action], axis=-1)

        return action, is_done

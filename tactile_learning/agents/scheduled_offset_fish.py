# Cleaner and FISH with manually scheduled offsets
# It should explore for a bit every once a while and then train with the 
# newly explored rollouts - not sure if the model would output 
# anything... 

import datetime
import glob
import os
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import sys

from pathlib import Path

from torchvision import transforms as T

from tactile_learning.models import *
from tactile_learning.utils import *
from tactile_learning.tactile_data import *
from holobot.robot.allegro.allegro_kdl import AllegroKDL

class ScheduledOffsetFISH: 
    def __init__(self,
        data_path, expert_demo_nums, expert_id, exploration,
        expert_frame_matches, episode_frame_matches,
        image_out_dir, tactile_out_dir, image_model_type, tactile_model_type,  
        reward_representations, policy_representations, features_repeat, view_num,   
        device, lr, feature_dim, hidden_dim, cricit_target_tau, 
        num_expl_steps, update_every_steps, stddev_schedule, stddev_clip,
        update_target_every, arm_offset_scale_factor, hand_offset_scale_factor, 
        offset_mask # Should be added to offset scheduler
    ):
        
        # rewards, ssim_base_factor, sinkhorn_rew_scale, auto_rew_scale, auto_rew_scale_factor
        # these parts should go to the rewarder module
        # bc weight_type / bc_weight_schedule is removed

        self.device = device 
        self.lr = lr 
        self.critic_target_tau = cricit_target_tau
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.update_target_every = update_target_every
        self.arm_offset_scale_factor = arm_offset_scale_factor
        self.hand_offset_scale_factor= hand_offset_scale_factor
        self.features_repeat = features_repeat

        self.data_path = data_path
        self.expert_id = expert_id
        self.expert_frame_matches = expert_frame_matches
        self.episode_frame_matches = episode_frame_matches
        self.exploration = exploration

        self.roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
        self.data = load_data(self.roots, demos_to_use=expert_demo_nums)
        
        _, self.image_encoder, self.image_transform  = init_encoder_info(self.device, image_out_dir, 'image', view_num=view_num, model_type=image_model_type)
        self._set_image_transform()
        image_repr_dim = 512

        tactile_cfg, self.tactile_encoder, _ = init_encoder_info(self.device, tactile_out_dir, 'tactile', view_num=view_num, model_type=tactile_model_type)
        tactile_img = TactileImage(
            tactile_image_size = tactile_cfg.tactile_image_size, 
            shuffle_type = None
        )
        tactile_repr_dim = tactile_cfg.encoder.tactile_encoder.out_dim if tactile_model_type == 'bc' else tactile_cfg.encoder.out_dim
        
        self.tactile_repr = TactileRepresentation( # This will be used when calculating the reward - not getting the observations
            encoder_out_dim = tactile_repr_dim,
            tactile_encoder = self.tactile_encoder,
            tactile_image = tactile_img,
            representation_type = 'tdex'
        )

        self.reward_representations = reward_representations
        self.policy_representations = policy_representations
        self.view_num = view_num
        self.inv_image_transform = get_inverse_image_norm() # This is only to be able to 

        # Freeze the encoders
        self.image_encoder.eval()
        self.tactile_encoder.eval()
        for param in self.image_encoder.parameters():
            param.requires_grad = False 
        for param in self.tactile_encoder.parameters():
            param.requires_grad = False

        # Set the expert_demo - it will set the tactile representations and image observations and save them in a dictionary
        self._set_expert_demos()

        repr_dim = 0
        if 'tactile' in policy_representations:
            repr_dim += tactile_repr_dim
        if 'image' in policy_representations:
            repr_dim += image_repr_dim
        if 'features' in policy_representations:
            repr_dim += 23 * features_repeat

        action_shape = 23
        self.offset_mask = torch.IntTensor(offset_mask).to(self.device)
        self.actor = Actor(repr_dim, action_shape, feature_dim,
                            hidden_dim, offset_mask).to(device)

        self.critic = Critic(repr_dim, action_shape, feature_dim,
                                hidden_dim).to(device)
        self.critic_target = Critic(repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
            
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Data augmentation
        self.image_aug = RandomShiftsAug(pad=4) # This is augmentation for the image
        self.image_normalize = T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)

        self.train()
        self.critic_target.train()

    def initialize_base_policy(self, base_policy_cfg):
        self.base_policy = hydra.utils.instantiate(
            base_policy_cfg, 
            expert_demos = self.expert_demos, 
            tactile_repr_size = self.tactile_repr.size
        )

    def initialize_rewarder(self, rewarder_cfg):
        self.rewarder = hydra.utils.instantiate(
            rewarder_cfg,
        )

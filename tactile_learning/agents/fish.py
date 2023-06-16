# Agent implementation in fish
"""
This takes potil_vinn_offset and compute q-filter on encoder_vinn and vinn_action_qfilter
"""
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

class FISHAgent:
    def __init__(self,
        data_path, expert_demo_nums, expert_id, reward_matching_steps, mock_demo_nums,
        features_repeat, sum_experts, experiment_name, end_frames_repeat,
        scale_representations, exponential_exploration,
        exponential_offset_exploration, match_from_both, expert_frame_matches,
        episode_frame_matches, ssim_base_factor, exploration,
        vinn_image_out_dir, vinn_image_model_type, max_vinn_steps,
        image_out_dir, tactile_out_dir, image_model_type, tactile_model_type, # This is used to get the tactile representation size
        reward_representations, policy_representations, view_num,
        action_shape, device, lr, feature_dim,
        hidden_dim, critic_target_tau, num_expl_steps,
        update_every_steps, stddev_schedule, stddev_clip, augment,
        rewards, sinkhorn_rew_scale, update_target_every,
        auto_rew_scale, auto_rew_scale_factor,
        bc_weight_type, bc_weight_schedule, arm_offset_scale_factor, hand_offset_scale_factor, offset_mask
    ):
        
        self.device = device
        self.lr = lr
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        # self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.augment = augment
        self.rewards = rewards
        self.sinkhorn_rew_scale = sinkhorn_rew_scale
        self.update_target_every = update_target_every
        self.auto_rew_scale = auto_rew_scale
        self.auto_rew_scale_factor = auto_rew_scale_factor

        self.bc_weight_type = bc_weight_type
        self.bc_weight_schedule = bc_weight_schedule
        self.arm_offset_scale_factor = arm_offset_scale_factor
        self.hand_offset_scale_factor= hand_offset_scale_factor

        # NOTE: What is the q-filter mentioned?
        # NOTE: We might need to lower the representation dimenstion at some point

        # TODO: Load the data for that one demonstration
        self.data_path = data_path
        self.roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
        self.data = load_data(self.roots, demos_to_use=expert_demo_nums)
        self.expert_id = expert_id
        self.reward_matching_steps = reward_matching_steps
        self.features_repeat = features_repeat
        self.sum_experts = sum_experts
        self.experiment_name = experiment_name
        self.end_frames_repeat = end_frames_repeat
        self.scale_representations = scale_representations
        self.exponential_exploration = exponential_exploration
        # If expo offset is set to true we will be using additive offsets
        # for each timestep and the added offset will always be lower and 
        # lower as the time passes
        self.exponential_offset_exploration = exponential_offset_exploration
        self.last_offset = None # This will be changed if expo offset is set to true
        self.match_from_both = match_from_both
        self.expert_frame_matches = expert_frame_matches
        self.episode_frame_matches = episode_frame_matches
        self.ssim_base_factor = ssim_base_factor
        # self.base_policy_cfg = base_policy_cfg
        self.exploration = exploration
        
        # if base_policy == 'vinn_openloop':
        #     self.first_frame_encoder = resnet18(pretrained=True, out_dim=512) # NOTE: Set this differently
        # elif base_policy == 'vinn': # We'll need to load another encoder for getting the neigbors
        #    _, self.vinn_encoder, _ = init_encoder_info(
        #        self.device,
        #        out_dir = vinn_image_out_dir,
        #        encoder_type = 'image',
        #        model_type = vinn_image_model_type)
        #    self.max_vinn_steps = max_vinn_steps

        # Set the mock data
        self.mock_data = load_data(self.roots, demos_to_use=mock_demo_nums) # TODO: Delete this
        self.kdl_solver = AllegroKDL()

        # TODO: Load the encoders - both the normal ones and the target ones
        image_cfg, self.image_encoder, self.image_transform  = init_encoder_info(self.device, image_out_dir, 'image', view_num=view_num, model_type=image_model_type)
        self._set_image_transform()

        print('image_cfg.encoder: {}'.format(image_cfg.encoder))
        # image_repr_dim = image_cfg.encoder.image_encoder.out_dim if image_model_type == 'bc' else image_cfg.encoder.out_dim # TODO: Merge these in a better hydra config
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
        self.action_shape = action_shape
        self._set_expert_demos()

        repr_dim = 0
        if 'tactile' in policy_representations:
            repr_dim += tactile_repr_dim
        if 'image' in policy_representations:
            repr_dim += image_repr_dim
        if 'features' in policy_representations:
            repr_dim += 23 * features_repeat

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

        # Current step in episode
        self.count = 0
        # Openloop tracker
        self.curr_step = 0

        # # If the base policy is vinn should get all the representations first
        # if base_policy == 'vinn':
        #     self._get_all_representations()
        #     self.nn_k = 10 # We set these to what works for now - it never becomes more than 10 in our tasks
        #     self.buffer = NearestNeighborBuffer(10)
        #     self.knn = ScaledKNearestNeighbors(
        #         self.all_representations, # Both the input and the output of the nearest neighbors are
        #         self.all_representations,
        #         ['image', 'tactile'],
        #         [1, 1], # Could have tactile doubled
        #         self.tactile_repr.size
        #     )

        # elif base_policy == 'vinn_openloop':
        #     self.first_frame_img_encoder = resnet18(pretrained=True, out_dim=512).to(self.device).eval()
        #     self._get_first_frame_exp_representations()
        # Initialize and start the base policy
        # print('BASE POLCY CFG : {}, expert_demos: {}'.format(
        #     base_policy_cfg, self.expert_demos))
        # base_policy_cfg = OmegaConf.create(base_policy_cfg)
    #     self.base_policy_cfg = base_policy_cfg

    def initialize_base_policy(self, base_policy_cfg): 
        self.base_policy = hydra.utils.instantiate(
            base_policy_cfg,
            expert_demos = self.expert_demos,
            tactile_repr_size = self.tactile_repr.size,
        )

    def __repr__(self):
        return "fish_agent"

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def _set_image_transform(self):
        image_act_transform_arr = []
        if self.augment:
            image_act_transform_arr.append(
                RandomShiftsAug(pad=4)
            )
        image_act_transform_arr.append(
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)
        )

        self.image_act_transform = T.Compose(image_act_transform_arr)

        self.image_normalize = T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)

    def _set_expert_demos(self): # Will stack the end frames back to back
        # We'll stack the tactile repr and the image observations
        self.expert_demos = []
        image_obs = [] 
        tactile_reprs = []
        actions = []
        old_demo_id = -1
        for step_id in range(len(self.data['image']['indices'])): 
            # Set observations
            demo_id, tactile_id = self.data['tactile']['indices'][step_id]
            if (demo_id != old_demo_id and step_id > 0) or (step_id == len(self.data['image']['indices'])-1):
                
                if self.end_frames_repeat > 1:
                    # Stack the frame previous to the end - the end frame is not good - the last frame is the first frame...
                    frame_id_to_stack= -1 # Get the -2nd frame to stack - NOTE: This is a hack for the dataset - the end frame could be noisy sometimes and the last frame shouldn't be added!
                    image_to_append, tactile_to_append, action_to_append = image_obs[frame_id_to_stack], tactile_reprs[frame_id_to_stack], actions[frame_id_to_stack]                
                    image_obs = image_obs[:frame_id_to_stack]
                    tactile_reprs = tactile_reprs[:frame_id_to_stack]
                    actions = actions[:frame_id_to_stack]
                    for i in range(self.end_frames_repeat):
                        image_obs.append(image_to_append)
                        tactile_reprs.append(tactile_to_append)
                        actions.append(action_to_append)
                
                self.expert_demos.append(dict(
                    image_obs = torch.stack(image_obs, 0), # NOTE: I don't think there is a problem here 
                    tactile_repr = torch.stack(tactile_reprs, 0),
                    actions = np.stack(actions, 0)
                ))
                image_obs = [] 
                tactile_reprs = []
                actions = []

            tactile_value = self.data['tactile']['values'][demo_id][tactile_id]
            tactile_repr = self.tactile_repr.get(tactile_value, detach=False)

            _, image_id = self.data['image']['indices'][step_id]
            image = load_dataset_image(
                data_path = self.data_path, 
                demo_id = demo_id, 
                image_id = image_id,
                view_num = self.view_num,
                transform = self.image_transform
            )

            # Set actions
            _, allegro_action_id = self.data['allegro_actions']['indices'][step_id]
            allegro_joint_action = self.data['allegro_actions']['values'][demo_id][allegro_action_id]
            if self.action_shape == 19:
                allegro_action = self.kdl_solver.get_fingertip_coords(allegro_joint_action)
            else:
                allegro_action = allegro_joint_action 

            # Set kinova action 
            _, kinova_id = self.data['kinova']['indices'][step_id]
            kinova_action = self.data['kinova']['values'][demo_id][kinova_id]
            demo_action = np.concatenate([allegro_action, kinova_action], axis=-1)


            image_obs.append(image)
            tactile_reprs.append(tactile_repr)
            actions.append(demo_action)

            old_demo_id = demo_id
        
    # def _get_all_representations(self):
    #     print('Getting all representations')
    #     all_representations = []

    #     pbar = tqdm(total=len(self.data['tactile']['indices']))
    #     for index in range(len(self.data['tactile']['indices'])):
    #         # Get the representation data
    #         demo_id, tactile_id = self.data['tactile']['indices'][index]
    #         tactile_value = self.data['tactile']['values'][demo_id][tactile_id]
    #         tactile_repr = self.tactile_repr.get(tactile_value, detach=False)

    #         _, image_id = self.data['image']['indices'][index]
    #         image = load_dataset_image(
    #             data_path = self.data_path, 
    #             demo_id = demo_id, 
    #             image_id = image_id,
    #             view_num = self.view_num,
    #             transform = self.image_transform
    #         )
    #         image_repr = self.vinn_encoder(image.unsqueeze(0).to(self.device)).squeeze()
    #         all_repr = torch.concat([image_repr, tactile_repr], dim=-1).detach().cpu()
    #         all_representations.append(all_repr)

    #         pbar.update(1)

    #     pbar.close()
    #     self.all_representations = torch.stack(all_representations, dim=0)
    #     print('all_representations.shape: {}'.format(self.all_representations.shape))

    # def _get_vinn_action(self, obs, episode_step):
    #     # Get the current representation
    #     image_obs = obs['image_obs'].unsqueeze(0) / 255.
    #     tactile_repr = obs['tactile_repr'].numpy()
    #     image_obs = self.image_normalize(image_obs.float()).to(self.device)
    #     image_repr = self.vinn_encoder(image_obs).detach().cpu().numpy().squeeze()
    #     curr_repr = np.concatenate([image_repr, tactile_repr], axis=0)

    #     print('curr_repr.shape in _get_vinn_action: {}'.format(curr_repr.shape))

    #     # Choose the action with the buffer 
    #     _, nn_idxs, _ = self.knn.get_k_nearest_neighbors(curr_repr, k=self.nn_k)
    #     id_of_nn = self.buffer.choose(nn_idxs)
    #     nn_id = nn_idxs[id_of_nn]
    #     if nn_id+1 >= len(self.data['allegro_actions']['indices']): # If the chosen action is the action after the last action
    #         nn_idxs = np.delete(nn_idxs, id_of_nn)
    #         id_of_nn = self.buffer.choose(nn_idxs)
    #         nn_id = nn_idxs[id_of_nn]
    #     # Check if the closest neighbor is the last frame
    #     is_done = False
    #     curr_demo_id, _ = self.data['allegro_actions']['indices'][nn_id]
    #     next_demo_id, _ = self.data['allegro_actions']['indices'][nn_id+1]
    #     if next_demo_id != curr_demo_id:
    #         is_done = True
    #         nn_id -= 1 # Just send the last frame action
    #     elif episode_step > self.max_vinn_steps: 
    #         is_done = True

    #     demo_id, action_id = self.data['allegro_actions']['indices'][nn_id+1]  # Get the next commanded action (commanded actions are saved in that timestamp)
    #     nn_allegro_action = self.data['allegro_actions']['values'][demo_id][action_id]
    #     _, kinova_id = self.data['kinova']['indices'][nn_id+1] # Get the next kinova state (which is for kinova robot the same as the next commanded action)
    #     nn_kinova_action = self.data['kinova']['values'][demo_id][kinova_id]

    #     print('EPISODE STEP: {} self.max_vinn_steps: {} ID OF NN: {} NEAREST NEIGHBOR DEMO ID: {}, IS DONE IN VINN: {}'.format(
    #         episode_step, self.max_vinn_steps, id_of_nn, demo_id, is_done))

    #     action = np.concatenate([nn_allegro_action, nn_kinova_action], axis=-1)

    #     return action, is_done

    # def _get_first_frame_exp_representations(self): # This will be used for VINN
        
    #     exp_representations = [] 
    #     for expert_id in range(len(self.expert_demos)):
    #         first_frame_exp_obs = self.expert_demos[expert_id]['image_obs'][0:1,:].to(self.device)
    #         plt.imshow(np.transpose(first_frame_exp_obs[0].detach().cpu().numpy(), (1,2,0)))
    #         plt.savefig(f'expert_id_{expert_id}.png')
    #         print(f'expert_{expert_id} mean: {first_frame_exp_obs[0].mean()}')

    #         # NOTE: Expert demos should already be normalized
    #         first_frame_exp_representation = self.first_frame_img_encoder(first_frame_exp_obs)
    #         exp_representations.append(first_frame_exp_representation.detach().cpu().numpy().squeeze())

    #     self.exp_first_frame_reprs = np.stack(exp_representations, axis=0)
    #     print('self.exp_reprs.shape: {}'.format(self.exp_first_frame_reprs.shape))

    def _check_limits(self, offset_action):
        # limits = [-0.1, 0.1]
        hand_limits = [-self.hand_offset_scale_factor-0.2, self.hand_offset_scale_factor+0.2] 
        arm_limits = [-self.arm_offset_scale_factor-0.02, self.arm_offset_scale_factor+0.02]
        offset_action[:,:-7] = torch.clamp(offset_action[:,:-7], min=hand_limits[0], max=hand_limits[1])
        offset_action[:,-7:] = torch.clamp(offset_action[:,-7:], min=arm_limits[0], max=arm_limits[1])
        return offset_action

    # def _get_closest_expert_id(self, obs):
    #     # Get the representation of the current observation
    #     image_transform = T.Compose([
    #         T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)
    #     ])
        
    #     image_obs = image_transform(obs['image_obs'] / 255.).unsqueeze(0).to(self.device)
    #     plt.imshow(np.transpose(image_obs[0].detach().cpu().numpy(), (1,2,0)))
    #     plt.savefig(f'first_obs.png')
    #     curr_repr = self.first_frame_img_encoder(image_obs).detach().cpu().numpy().squeeze()

    #     # Get the distance of the curr_repr to the expert representaitons of the first frame
    #     l1_distances = self.exp_first_frame_reprs - curr_repr
    #     l2_distances = np.linalg.norm(l1_distances, axis=1)
    #     print('l2_distances.shape: {} - should be NUM_EXPERTS'.format(l2_distances.shape))
    #     sorted_idxs = np.argsort(l2_distances)

    #     print('SORTED EXPERTS: {}'.format(sorted_idxs))
    #     return sorted_idxs[0], l2_distances

    # Will give the next action in the step
    def base_act(self, obs, episode_step): # Returns the action for the base policy - openloop
        # TODO: You should get the nearest neighbor at the beginning and maybe at each step as well?
        if episode_step == 0:
            # Set the exploration
            if self.exploration == 'ou_noise':
                self.ou_noise = OrnsteinUhlenbeckActionNoise(
                    mu = np.zeros(len(self.offset_mask)), # The mean of the offsets should be 0
                    sigma = 0.8 # It will give bw -1 and 1 - then this gets multiplied by the scale factors ...
                )

            # if self.base_policy == 'vinn_openloop': # Get the expert id that has the closest representaiton in the beginning
            #     self.expert_id, self.l2_distances = self._get_closest_expert_id(obs)

        # Use expert_demos for base action retrieval
        # is_done = False
        # if episode_step >= len(self.expert_demos[self.expert_id]['actions']):
        #     episode_step = len(self.expert_demos[self.expert_id]['actions'])-1
        #     is_done = True

        # print('episode step: {}, len(self.expert_demos[self.expert_id][actions]: {}'.format(
        #     episode_step, len(self.expert_demos[self.expert_id]['actions'])
        # ))

        # if self.base_policy == 'vinn':
        #     action, is_done = self._get_vinn_action(obs, episode_step)
        # else:
        #     action = self.expert_demos[self.expert_id]['actions'][episode_step]

        action, is_done = self.base_policy.act( # TODO: Make sure these are good
            obs, episode_step
        )

        return torch.FloatTensor(action).to(self.device).unsqueeze(0), is_done


    def act(self, obs, global_step, episode_step, eval_mode, metrics=None):
        with torch.no_grad():
            base_action, is_done = self.base_act(obs, episode_step)

        print('base_action.shape: {}'.format(base_action.shape))

        with torch.no_grad():
            # Get the action image_obs
            obs = self._get_policy_reprs_from_obs( # This method is called with torch.no_grad() in training anyways
                image_obs = obs['image_obs'].unsqueeze(0) / 255.,
                tactile_repr = obs['tactile_repr'].unsqueeze(0),
                features = obs['features'].unsqueeze(0),
                representation_types=self.policy_representations
            )

        print('obs.shape: {} in act'.format(obs.shape))

        stddev = schedule(self.stddev_schedule, global_step)
        dist = self.actor(obs, base_action, stddev)
        if eval_mode:
            offset_action = dist.mean
        else:
            offset_action = dist.sample(clip=None)

            # Exploration
            if self.exponential_exploration:
                epsilon = exponential_epsilon_decay(
                    step_idx=global_step,
                    epsilon_decay=self.num_expl_steps
                )
                rand_num = random.random()
                if rand_num < epsilon:
                    print('EXPLORING!!!')
                    offset_action.uniform_(-1.0, 1.0)
            elif self.exploration == 'ou_noise':
                if global_step < self.num_expl_steps:
                    offset_action = torch.FloatTensor(self.ou_noise()).to(self.device).unsqueeze(0)
                    print('ou_noise offset: {}'.format(offset_action.shape))
            else:
                if global_step < self.num_expl_steps:
                    offset_action.uniform_(-1.0, 1.0)

        offset_action *= self.offset_mask 
        is_explore = global_step < self.num_expl_steps or (self.exponential_exploration and rand_num < epsilon)

        if is_explore and self.exponential_offset_exploration:  # TODO: Do the OU Noise
            offset_action[:,:-7] *= self.hand_offset_scale_factor / ((episode_step+1)/2) 
            offset_action[:,-7:] *= self.arm_offset_scale_factor / ((episode_step+1)/2)
            if episode_step > 0:
                offset_action += self.last_offset
            self.last_offset = offset_action
        else:
            offset_action[:,:-7] *= self.hand_offset_scale_factor
            offset_action[:,-7:] *= self.arm_offset_scale_factor

        # Check if the offset action is higher than the limits
        offset_action = self._check_limits(offset_action)

        print('HAND OFFSET ACTION: {}'.format(
            offset_action[:,:-7]
        ))
        print('ARM OFFSET ACTION: {}'.format(
            offset_action[:,-7:]
        ))

        action = base_action + offset_action

        # If metrics are not None then plot the offsets
        metrics = dict()
        for i in range(len(self.offset_mask)):
            if self.offset_mask[i] == 1: # Only log the times when there is an allowed offset
                if eval_mode:
                    offset_key = f'offset_{i}_eval'
                else:
                    offset_key = f'offset_{i}_train'
                metrics[offset_key] = offset_action[:,i]

        return action.cpu().numpy()[0], base_action.cpu().numpy()[0], is_done, metrics

    def _find_closest_mock_step(self, curr_step):
        offset_step = 0
        if curr_step == 0:
            return curr_step

        curr_demo_id, _ = self.mock_data['allegro_actions']['indices'][curr_step]
        next_demo_id, _ = self.mock_data['allegro_actions']['indices'][curr_step+offset_step]
        while curr_demo_id == next_demo_id:
            offset_step += 1
            next_demo_id, _ = self.mock_data['allegro_actions']['indices'][(curr_step+offset_step) % len(self.mock_data['allegro_actions']['indices'])]
            

        next_demo_step = (curr_step+offset_step) % len(self.mock_data['allegro_actions']['indices'])
        return next_demo_step

    # Method that returns the next action in the mock data
    def mock_act(self, obs, step, max_step): # Returns the action for the base policy - TODO: This will be used after we have the environment
        if step > 0 and step % max_step == 0:
            self.curr_step = self._find_closest_mock_step(self.curr_step)
        
        # Get the action in the current step
        demo_id, action_id = self.mock_data['allegro_actions']['indices'][self.curr_step]
        allegro_joint_action = self.mock_data['allegro_actions']['values'][demo_id][action_id]
        if self.action_shape == 19:
            allegro_action = self.kdl_solver.get_fingertip_coords(allegro_joint_action)
        else:
            allegro_action = allegro_joint_action
        
        # Get the kinova action 
        _, kinova_id = self.mock_data['kinova']['indices'][self.curr_step]
        kinova_action = self.mock_data['kinova']['values'][demo_id][kinova_id]

        # Concatenate the actions 
        demo_action = np.concatenate([allegro_action, kinova_action], axis=-1)

        self.curr_step = (self.curr_step+1) % (len(self.mock_data['allegro_actions']['indices']))

        return demo_action, np.zeros(self.action_shape) # Base action will be 0s only for now

    def update_critic(self, obs, action, base_next_action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, base_next_action, stddev)

            offset_action = dist.sample(clip=self.stddev_clip)
            offset_action[:,:-7] *= self.hand_offset_scale_factor
            offset_action[:,-7:] *= self.arm_offset_scale_factor 
            next_action = base_next_action + offset_action

            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)

        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize encoder and critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()
            
        return metrics

    def update_actor(self, obs, obs_expert, obs_qfilter, action_expert, base_action, base_action_expert, bc_regularize, step):
        metrics = dict()

        stddev = schedule(self.stddev_schedule, step)

        # compute action offset
        dist = self.actor(obs, base_action, stddev)
        action_offset = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action_offset).sum(-1, keepdim=True)

        # compute action
        action_offset[:-7] *= self.hand_offset_scale_factor
        action_offset[-7:] *= self.arm_offset_scale_factor 
        action = base_action + action_offset 
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        # Compute bc weight
        if not bc_regularize: # This will be false
            bc_weight = 0.0
        elif self.bc_weight_type == "linear":
            bc_weight = schedule(self.bc_weight_schedule, step)
        elif self.bc_weight_type == "qfilter": # NOTE: This is for ROT but in FISH they had it set to false
            """
            Soft Q-filtering inspired from 			
            Nair, Ashvin, et al. "Overcoming exploration in reinforcement 
            learning with demonstrations." 2018 IEEE international 
            conference on robotics and automation (ICRA). IEEE, 2018.
            """
            with torch.no_grad():
                stddev = 0.1
                action_qf = base_action.clone()
                Q1_qf, Q2_qf = self.critic(obs_qfilter.clone(), action_qf)
                Q_qf = torch.min(Q1_qf, Q2_qf)
                bc_weight = (Q_qf>Q).float().mean().detach()

        actor_loss = - Q.mean() * (1-bc_weight)

        if bc_regularize:
            stddev = 0.1
            dist_expert = self.actor(obs_expert, base_action_expert, stddev)
            action_expert_offset = dist_expert.sample(clip=self.stddev_clip)
            action_expert_offset[:-7] *= self.hand_offset_scale_factor
            action_expert_offset[-7:] *= self.arm_offset_scale_factor 
            # action_expert_offset = dist_expert.sample(clip=self.stddev_clip) * self.offset_scale_factor

            true_offset = torch.zeros(action_expert_offset.shape).to(self.device)
            log_prob_expert = dist_expert.log_prob(true_offset).sum(-1, keepdim=True)
            actor_loss += - log_prob_expert.mean()*bc_weight*0.03

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        
        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_logprob'] = log_prob.mean().item()
        metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
        metrics['actor_q'] = Q.mean().item()
        if bc_regularize and self.bc_weight_type == "qfilter":
            metrics['actor_qf'] = Q_qf.mean().item()
        metrics['bc_weight'] = bc_weight
        metrics['regularized_rl_loss'] = -Q.mean().item()* (1-bc_weight)
        metrics['rl_loss'] = -Q.mean().item()
        if bc_regularize:
            metrics['regularized_bc_loss'] = - log_prob_expert.mean().item()*bc_weight*0.03
            metrics['bc_loss'] = - log_prob_expert.mean().item()*0.03
            
        return metrics

    def _get_policy_reprs_from_obs(self, image_obs, tactile_repr, features, representation_types):
         # Get the representations
        reprs = []
        if 'image' in representation_types:
            # Current representations
            image_obs = self.image_act_transform(image_obs.float()).to(self.device)
            image_reprs = self.image_encoder(image_obs)
            reprs.append(image_reprs)

        if 'tactile' in representation_types:
            tactile_reprs = tactile_repr.to(self.device) # This will give all the representations of one batch
            reprs.append(tactile_reprs)

        if 'features' in representation_types:
            repeated_features = features.repeat(1, self.features_repeat)
            reprs.append(repeated_features.to(self.device))

        return torch.concat(reprs, axis=-1) # Concatenate the representations to get the final representations

    def update(self, replay_iter, step, bc_regularize=False, expert_replay_iter=None, ssl_replay_iter=None):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        image_obs, tactile_repr, features, action, base_action, reward, discount, next_image_obs, next_tactile_repr, next_features, base_next_action = to_torch(
            batch, self.device)
        
        # Multiply action with the offset mask just incase if the buffer was not saved that way
        # action *= self.offset_mask
        offset_action = action - base_action
        offset_action *= self.offset_mask 
        action = base_action + offset_action
        print('UPDATE - image_obs.shape: {}'.format(image_obs.shape))

        # Get the representations
        obs = self._get_policy_reprs_from_obs(
            image_obs = image_obs, # These are stacked PIL images?
            tactile_repr = tactile_repr,
            features = features,
            representation_types=self.policy_representations
        )
        next_obs = self._get_policy_reprs_from_obs(
            image_obs = next_image_obs, 
            tactile_repr = next_tactile_repr,
            features = next_features,
            representation_types=self.policy_representations
        )

        print('obs.shape: {}, next_obs.shape: {}'.format(
        	obs.shape, next_obs.shape
        ))

        obs_qfilter = None
        obs_expert = None 
        action_expert = None
        base_action_expert = None

        metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, base_next_action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), obs_expert, obs_qfilter, action_expert, base_action, base_action_expert, bc_regularize, step))

        # update critic target
        soft_update_params(self.critic, self.critic_target,
                                    self.critic_target_tau)

        return metrics

    def ot_rewarder(self, episode_obs, episode_id, mock=False, visualize=False, exponential_weight_init=False): # TODO: Delete the mock option
        
        # NOTE: In this code we're not using target encoder since the encoders are already frozen

        all_ot_rewards = []
        cost_matrices = []
        best_reward_sum = - sys.maxsize
        best_ot_reward_id = -1

        # Get the episode representations
        curr_reprs = []
        if 'image' in self.reward_representations: # We will not be using features for reward for sure
            if (not self.expert_frame_matches is None) and (not self.episode_frame_matches is None):
                image_reprs = self.image_encoder(episode_obs['image_obs'][-self.episode_frame_matches:,:].to(self.device))
            else:
                if self.match_from_both:
                    image_reprs = self.image_encoder(episode_obs['image_obs'][-self.reward_matching_steps:,:].to(self.device))
                else:
                    image_reprs = self.image_encoder(episode_obs['image_obs'].to(self.device))
            curr_reprs.append(image_reprs) # NOTE: We should alwasys get all of the observation

        if 'tactile' in self.reward_representations:
            if (not self.expert_frame_matches is None) and (not self.episode_frame_matches is None):
                tactile_reprs = episode_obs['tactile_repr'][-self.episode_frame_matches:,:].to(self.device)
            else:
                if self.match_from_both:
                    tactile_reprs = episode_obs['tactile_repr'][-self.reward_matching_steps:,:].to(self.device) # This will give all the representations of one episode
                else:
                    tactile_reprs = episode_obs['tactile_repr'].to(self.device)
            curr_reprs.append(tactile_reprs)

        # Concatenate everything now
        obs = torch.concat(curr_reprs, dim=-1).detach()
        print('obs.shape in ot_rewarder: {}'.format(obs.shape))

        for expert_id in range(len(self.expert_demos)):
            # Get the expert representations
            exp_reprs = []
            if 'image' in self.reward_representations: # We will not be using features for reward for sure
                if (not self.expert_frame_matches is None) and (not self.episode_frame_matches is None):
                    expert_image_reprs = self.image_encoder(self.expert_demos[expert_id]['image_obs'][-self.expert_frame_matches:,:].to(self.device))
                else:
                    expert_image_reprs = self.image_encoder(self.expert_demos[expert_id]['image_obs'][-self.reward_matching_steps:,:].to(self.device))
                exp_reprs.append(expert_image_reprs)

            if 'tactile' in self.reward_representations:
                if (not self.expert_frame_matches is None) and (not self.episode_frame_matches is None):
                    expert_tactile_reprs = self.expert_demos[expert_id]['tactile_repr'][-self.expert_frame_matches:,:].to(self.device)
                else:
                    expert_tactile_reprs = self.expert_demos[expert_id]['tactile_repr'][-self.reward_matching_steps:,:].to(self.device)
                exp_reprs.append(expert_tactile_reprs)

            # Concatenate everything now
            exp = torch.concat(exp_reprs, dim=-1).detach()
            print('exp[{}].shape in ot_rewarder : {}'.format(expert_id, exp.shape))

            if self.rewards == 'sinkhorn_cosine':
                cost_matrix = cosine_distance(
                        obs, exp)  # Get cost matrix for samples using critic network.
                transport_plan = optimal_transport_plan(
                    obs, exp, cost_matrix, method='sinkhorn',
                    niter=100, exponential_weight_init=exponential_weight_init).float()  # Getting optimal coupling
                ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
                    torch.mm(transport_plan,
                                cost_matrix.T)).detach().cpu().numpy()
                
            elif self.rewards == 'sinkhorn_euclidean':
                cost_matrix = euclidean_distance(
                    obs, exp)  # Get cost matrix for samples using critic network.
                transport_plan = optimal_transport_plan(
                    obs, exp, cost_matrix, method='sinkhorn',
                    niter=100, exponential_weight_init=exponential_weight_init).float()  # Getting optimal coupling
                ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
                    torch.mm(transport_plan,
                                cost_matrix.T)).detach().cpu().numpy()
                
            elif self.rewards == 'cosine':
                exp = torch.cat((exp, exp[-1].unsqueeze(0)))
                ot_rewards = -(1. - F.cosine_similarity(obs, exp))
                ot_rewards *= self.sinkhorn_rew_scale
                ot_rewards = ot_rewards.detach().cpu().numpy()
                
            elif self.rewards == 'euclidean':
                exp = torch.cat((exp, exp[-1].unsqueeze(0)))
                ot_rewards = -(obs - exp).norm(dim=1)
                ot_rewards *= self.sinkhorn_rew_scale
                ot_rewards = ot_rewards.detach().cpu().numpy()

            elif self.rewards == 'ssim': # Use structural similarity index match
                if (not self.expert_frame_matches is None) and (not self.episode_frame_matches is None):
                    episode_img = self.inv_image_transform(episode_obs['image_obs'][-self.episode_frame_matches:,:]) 
                    expert_img = self.inv_image_transform(self.expert_demos[expert_id]['image_obs'][-self.expert_frame_matches:,:])
                else:
                    if self.match_from_both:
                        episode_img = self.inv_image_transform(episode_obs['image_obs'][-self.reward_matching_steps:,:]) 
                    else:
                        episode_img = self.inv_image_transform(episode_obs['image_obs'])
                    expert_img = self.inv_image_transform(self.expert_demos[expert_id]['image_obs'][-self.reward_matching_steps:,:])

                ot_rewards = structural_similarity_index(
                    x = expert_img,
                    y = episode_img,
                    base_factor = self.ssim_base_factor
                )
                # ot_rewards -= self.ssim_base_factor 
                # ot_rewards *= self.sinkhorn_rew_scale / (1 - self.ssim_base_factor) # We again scale it to 10 in the beginning
                ot_rewards *= self.sinkhorn_rew_scale
                cost_matrix = torch.FloatTensor(ot_rewards)
                
            else:
                raise NotImplementedError()
            
            all_ot_rewards.append(ot_rewards)
            sum_ot_rewards = np.sum(ot_rewards) 
            cost_matrices.append(cost_matrix)
            if sum_ot_rewards > best_reward_sum:
                best_reward_sum = sum_ot_rewards
                best_ot_reward_id = expert_id
        
        if self.sum_experts:
            final_cost_matrix = sum(cost_matrices) # Sum it through all of the cost matrices
            final_ot_reward = sum(all_ot_rewards)
        else:
            final_cost_matrix = cost_matrices[best_ot_reward_id]
            final_ot_reward = all_ot_rewards[best_ot_reward_id]

        print('all_ot_rewards: {}'.format(all_ot_rewards))
        print('final_ot_reward: {}'.format(final_ot_reward))

        if visualize:
            self.plot_cost_matrix(final_cost_matrix, expert_id=best_ot_reward_id, episode_id=episode_id)

        return final_ot_reward
    
    def plot_cost_matrix(self, cost_matrix, expert_id, episode_id, file_name=None):
        if file_name is None:
            ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            file_name = f'{ts}_reward_{self.reward_representations}_expert_{expert_id}_ep_{episode_id}_cost_matrix.png'

        # Plot MxN matrix if file_name is given -> it will save the plot there if so
        cost_matrix = cost_matrix.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(15,15),nrows=1,ncols=1)
        im = ax.matshow(cost_matrix)
        ax.set_title(f'File: {file_name}')
        fig.colorbar(im, ax=ax, label='Interactive colorbar')

        plt.xlabel('Expert Demo Timesteps')
        plt.ylabel('Observation Timesteps')
        plt.title(file_name)

        dump_dir = Path('/home/irmak/Workspace/tactile-learning/online_training_outs/costs') / self.experiment_name
        # dump_dir = os.path.join('/home/irmak/Workspace/tactile-learning/train_video/costs/{}')
        os.makedirs(dump_dir, exist_ok=True)
        dump_file = os.path.join(dump_dir, file_name)
        plt.savefig(dump_file, bbox_inches='tight')
        plt.close()   

    def save_snapshot(self):
        keys_to_save = ['actor', 'critic', 'image_encoder']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        return payload

    def load_snapshot(self, payload):
        loaded_encoder = None
        for k, v in payload.items():
            print('k: {}'.format(k))
            if k == 'image_encoder':
                loaded_encoder = v

        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.image_encoder.load_state_dict(loaded_encoder.state_dict()) # NOTE: In the actual repo they use self.vinn_encoder rather than loaded_encoder 

        self.image_encoder.eval()
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.policy.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def load_snapshot_eval(self, payload, bc=False):
        for k, v in payload.items():
            self.__dict__[k] = v
        # NOTE: In the FISH-main code we're loading the state dictionary for the encoder as well
        # here we're freezing the encoder - so not necessary
        self.critic_target.load_state_dict(self.critic.state_dict()) 
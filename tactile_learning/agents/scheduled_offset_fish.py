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

    def __repr__(self):
        return 'offset_scheduled_fish_agent'

    def initialize_modules(self, base_policy_cfg, rewarder_cfg, explorer_cfg): 
        self.base_policy = hydra.utils.instantiate(
            base_policy_cfg,
            expert_demos = self.expert_demos,
            tactile_repr_size = self.tactile_repr.size,
        )
        self.rewarder = hydra.utils.instantiate(
            rewarder_cfg,
            expert_demos = self.expert_demos, 
            image_encoder = self.image_encoder if 'image' in self.reward_representations else None,
            tactile_encoder = self.tactile_encoder if 'tactile' in self.reward_representations else None
        )
        self.explorer = hydra.utils.instantiate(
            explorer_cfg
        )

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def _set_image_transform(self):
        self.image_act_transform = T.Compose([
            RandomShiftsAug(pad=4),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)
        ])
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
            allegro_action = allegro_joint_action 

            # Set kinova action 
            _, kinova_id = self.data['kinova']['indices'][step_id]
            kinova_action = self.data['kinova']['values'][demo_id][kinova_id]
            demo_action = np.concatenate([allegro_action, kinova_action], axis=-1)

            image_obs.append(image)
            tactile_reprs.append(tactile_repr)
            actions.append(demo_action)

            old_demo_id = demo_id

    def _check_limits(self, offset_action):
        # limits = [-0.1, 0.1]
        hand_limits = [-self.hand_offset_scale_factor-0.2, self.hand_offset_scale_factor+0.2] 
        arm_limits = [-self.arm_offset_scale_factor-0.02, self.arm_offset_scale_factor+0.02]
        offset_action[:,:-7] = torch.clamp(offset_action[:,:-7], min=hand_limits[0], max=hand_limits[1])
        offset_action[:,-7:] = torch.clamp(offset_action[:,-7:], min=arm_limits[0], max=arm_limits[1])
        return offset_action

    # Will give the next action in the step
    def base_act(self, obs, episode_step): # Returns the action for the base policy - openloop
        action, is_done = self.base_policy.act( # TODO: Make sure these are good
            obs, episode_step
        )

        return torch.FloatTensor(action).to(self.device).unsqueeze(0), is_done

    def act(self, obs, global_step, episode_step, eval_mode, metrics=None):
        with torch.no_grad():
            base_action, is_done = self.base_act(obs, episode_step)

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

            offset_action = self.explorer.explore(
                offset_action = offset_action,
                global_step = global_step, 
                episode_step = episode_step,
                device = self.device
            )

        offset_action *= self.offset_mask 

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
        offset_action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(offset_action).sum(-1, keepdim=True)

        # compute action
        offset_action[:,:-7] *= self.hand_offset_scale_factor
        offset_action[:,-7:] *= self.arm_offset_scale_factor 
        action = base_action + offset_action 
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = - Q.mean()  # BC weight is not added to this

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        
        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_logprob'] = log_prob.mean().item()
        metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
        metrics['actor_q'] = Q.mean().item()
        metrics['rl_loss'] = -Q.mean().item()
            
        return metrics

    
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

    def ot_rewarder(self, episode_obs, episode_id, visualize=False, exponential_weight_init=False): # TODO: Delete the mock option

        final_reward, final_cost_matrix, best_expert_id = self.rewarder.get(
            obs = episode_obs
        )

        print('final_reward: {}'.format(final_reward))

        if visualize:
            self.plot_cost_matrix(final_cost_matrix, expert_id=best_expert_id, episode_id=episode_id)

        return final_reward
    
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
        self.critic_target.load_state_dict(self.critic.state_dict()) 
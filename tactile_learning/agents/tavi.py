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
import torch.nn.functional as F
import sys

from pathlib import Path

from torchvision import transforms as T

from tactile_learning.models import *
from tactile_learning.utils import *
from tactile_learning.tactile_data import *
from holobot.robot.allegro.allegro_kdl import AllegroKDL
from .agent import Agent

class TAVI(Agent):
    def __init__(self,
        data_path, expert_demo_nums, expert_id, # Agent parameters
        features_repeat, experiment_name, 
        image_out_dir, tactile_out_dir, image_model_type, tactile_model_type, # This is used to get the tactile representation size
        policy_representations, view_num,
        action_shape, device, lr, 
        feature_dim, hidden_dim, # TAVI parameters
        critic_target_tau, num_expl_steps,
        update_every_steps, stddev_schedule, stddev_clip, 
        arm_offset_scale_factor, hand_offset_scale_factor, offset_mask
    ):
        
        # Super Agent sets the encoders, transforms and expert demonstrations
        super().__init__(
            data_path = data_path,
            expert_demo_nums=expert_demo_nums,
            image_out_dir=image_out_dir, image_model_type=image_model_type,
            tactile_out_dir=tactile_out_dir, tactile_model_type=tactile_model_type,
            view_num=view_num, device=device, lr=lr, update_every_steps=update_every_steps,
            stddev_schedule=stddev_schedule, stddev_slip=stddev_clip, features_repeat=features_repeat,
            experiment_name=experiment_name
        )

        self.critic_target_tau = critic_target_tau
        self.num_expl_steps = num_expl_steps
        self.arm_offset_scale_factor = arm_offset_scale_factor
        self.hand_offset_scale_factor= hand_offset_scale_factor

        self.expert_id = expert_id

        # Set the models
        self.policy_representations = policy_representations
        repr_dim = self.repr_dim(type='policy')
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

        self.train()
        self.critic_target.train()

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

    def __repr__(self):
        return "fish_agent"

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

        print('base_action.shape: {}'.format(base_action.shape))

        with torch.no_grad():
            # Get the action image_obs
            obs = self._get_policy_reprs_from_obs( # This method is called with torch.no_grad() in training anyways
                image_obs = obs['image_obs'].unsqueeze(0) / 255.,
                tactile_repr = obs['tactile_repr'].unsqueeze(0),
                features = obs['features'].unsqueeze(0),
                representation_types=self.policy_representations
            )

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

    def update_actor(self, obs, base_action, step):
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
        actor_loss = - Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        
        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_logprob'] = log_prob.mean().item()
        metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
        metrics['actor_q'] = Q.mean().item()

        return metrics
    
    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        image_obs, tactile_repr, features, action, base_action, reward, discount, next_image_obs, next_tactile_repr, next_features, base_next_action = to_torch(
            batch, self.device)
        
        # Multiply action with the offset mask just incase if the buffer was not saved that way
        offset_action = action - base_action
        offset_action *= self.offset_mask 
        action = base_action + offset_action

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

        metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, base_next_action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), base_action, step))

        # update critic target
        soft_update_params(self.critic, self.critic_target,
                                    self.critic_target_tau)

        return metrics

    def get_reward(self, episode_obs, episode_id, visualize=False): # TODO: Delete the mock option
        
        final_reward, final_cost_matrix, best_expert_id = self.rewarder.get(
            obs = episode_obs
        )

        # Update the reward scale if it's the first episode
        if episode_id == 1:
            # Auto update the reward scale and get the rewards again
            self.rewarder.update_scale(current_rewards = final_reward)
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
        self.critic_target.load_state_dict(self.critic.state_dict()) 
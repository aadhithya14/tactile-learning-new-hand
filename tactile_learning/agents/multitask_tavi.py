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
from .multitask_agent import MultitaskAgent

class MultitaskTAVI(MultitaskAgent):
    def __init__(self,
        data_path, expert_demo_nums, # Agent parameters
        image_out_dir, image_model_type, # Encoders
        tactile_out_dir, tactile_model_type, 
        # policy_representations, 
        features_repeat, # Training parameters
        experiment_name, view_num, device, 
        action_shape, 
        # critic_target_tau, num_expl_steps,
        update_critic_frequency, update_critic_target_frequency, update_actor_frequency,
        # update_every_steps, update_critic_every_steps, update_actor_every_steps,
        # arm_offset_scale_factor, hand_offset_scale_factor, 
        offset_mask, # Task based offset parameters
        **kwargs
    ):
        
        # Super Agent sets the encoders, transforms and expert demonstrations
        super().__init__(
            data_path=data_path,
            expert_demo_nums=expert_demo_nums,
            image_out_dir=image_out_dir, image_model_type=image_model_type,
            tactile_out_dir=tactile_out_dir, tactile_model_type=tactile_model_type,
            view_num=view_num, device=device, update_critic_frequency=update_critic_frequency,
            features_repeat=features_repeat,
            experiment_name=experiment_name, update_critic_target_frequency=update_critic_target_frequency,
            update_actor_frequency=update_actor_frequency, **kwargs
        )

        #NOTE All of the parameters are getting set in the agent
        # self.critic_target_tau = critic_target_tau
        # self.num_expl_steps = num_expl_steps
        # self.arm_offset_scale_factor = arm_offset_scale_factor
        # self.hand_offset_scale_factor= hand_offset_scale_factor

        # # Set the models
        # self.policy_representations = policy_representations
        # repr_dim = self.repr_dim(type='policy')
        self.action_shape = action_shape
        self.offset_mask = torch.IntTensor(offset_mask).to(self.device)
        
        self.train()
        # self.actor = Actor(repr_dim, action_shape, feature_dim,
        #                     hidden_dim, offset_mask).to(device)

        # self.critic = Critic(repr_dim, action_shape, feature_dim,
        #                         hidden_dim).to(device)
        # self.critic_target = Critic(repr_dim, action_shape,
        #                             feature_dim, hidden_dim).to(device)
        # self.critic_target.load_state_dict(self.critic.state_dict())
            
        # self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        # self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # self.train()
        # self.critic_target.train()

        self.task_num = 0
        self.task_step = 0

    def initialize_modules(self, rl_learner_cfg, base_policy_cfg, rewarder_cfg, explorer_cfg): 
        rl_learner_cfg.action_shape = self.action_shape
        rl_learner_cfg.repr_dim = self.repr_dim(type='policy')
        self.rl_learner = hydra.utils.instantiate(
            rl_learner_cfg,
            actions_offset = True, 
            # repr_dim = self.repr_dim(type='policy'),
            # action_shape = self.action_shape
            hand_offset_scale_factor = self.hand_offset_scale_factor,
            arm_offset_scale_factor = self.arm_offset_scale_factor
        )

        # NOTE: base_policy and rewarder should be different in a sequence of tasks because of different expert_demos
        # explorer keeps the same?
        self.base_policy = []
        self.rewarder = []
        for task_num in range(len(self.expert_demos)): 
            self.base_policy.append(hydra.utils.instantiate(
                base_policy_cfg,
                expert_demos = self.expert_demos[task_num],
                tactile_repr_size = self.tactile_repr.size,
            ))
            self.rewarder.append(hydra.utils.instantiate(
                rewarder_cfg,
                expert_demos = self.expert_demos[task_num], 
                image_encoder = self.image_encoder[task_num],
                tactile_encoder = self.tactile_encoder
            ))
        self.explorer = hydra.utils.instantiate(
            explorer_cfg
        )

    def __repr__(self):
        return f"tavi_{repr(self.rl_learner)}"

    def _check_limits(self, offset_action):
        # limits = [-0.1, 0.1]
        hand_limits = [-self.hand_offset_scale_factor[self.task_num]-0.2, self.hand_offset_scale_factor[self.task_num]+0.2] 
        arm_limits = [-self.arm_offset_scale_factor[self.task_num]-0.02, self.arm_offset_scale_factor[self.task_num]+0.02]
        offset_action[:,:-7] = torch.clamp(offset_action[:,:-7], min=hand_limits[0], max=hand_limits[1])
        offset_action[:,-7:] = torch.clamp(offset_action[:,-7:], min=arm_limits[0], max=arm_limits[1])
        return offset_action

    # Will give the next action in the step
    def base_act(self, obs, episode_step): # Returns the action for the base policy - openloop

        action, is_done = self.base_policy[self.task_num].act( # TODO: Make sure these are good
            obs, episode_step
        )  
        return torch.FloatTensor(action).to(self.device).unsqueeze(0), is_done


    def act(self, obs, global_step, episode_step, eval_mode, metrics=None):

        with torch.no_grad():
            base_action, episode_is_done = self.base_act(obs, self.task_step)

        # #NOTE: If there is still successing tasks, then the episode is not done: 
        # if (self.task_num+1) < len(self.data):
        #     is_done = False
        # else: 
        #     is_done = True

        print('base_action.shape: {}'.format(base_action.shape))

        # if global_step > self.expert_demos[current_policy]['demo_length']:
        #     load_snapshot('next_policy')

        with torch.no_grad():
            # Get the action image_obs
            obs = self._get_policy_reprs_from_obs( # This method is called with torch.no_grad() in training anyways
                image_obs = obs['image_obs'].unsqueeze(0) / 255.,
                tactile_repr = obs['tactile_repr'].unsqueeze(0),
                features = obs['features'].unsqueeze(0),
                representation_types=self.policy_representations,
                task_num = self.task_num
            )

        # stddev = schedule(self.stddev_schedule, global_step) - NOTE: stddev_scheduling should be done inside the rl learner
        # dist = self.actor(obs, base_action, stddev)
        # if eval_mode:
        #     offset_action = dist.mean
        # else:
        #     offset_action = dist.sample(clip=None)

        #     offset_action = self.explorer.explore(
        #         offset_action = offset_action,
        #         global_step = global_step, 
        #         episode_step = episode_step,
        #         device = self.device
        #     )


        offset_action = self.rl_learner.act(
            obs=obs, eval_mode=eval_mode, base_action=base_action,
            global_step=global_step)
        
        offset_action = self.explorer.explore(
            offset_action = offset_action,
            global_step = global_step,
            episode_step = episode_step, 
            device = self.device,
            eval_mode = eval_mode
        )

        print('offset_action: {}'.format(offset_action))

        offset_action *= self.offset_mask[self.task_num]
        offset_action[:,:-7] *= self.hand_offset_scale_factor[self.task_num]
        offset_action[:,-7:] *= self.arm_offset_scale_factor[self.task_num]

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
        for i in range(len(self.offset_mask[self.task_num])):
            if self.offset_mask[self.task_num][i] == 1: # Only log the times when there is an allowed offset
                if eval_mode:
                    offset_key = f'offsets_eval/offset_{i}'
                else:
                    offset_key = f'offsets_train/offset_{i}'
                metrics[offset_key] = offset_action[:,i]
        
        #NOTE: if the episode is done, then task_num is reduced to 0
        is_done = False
        if episode_is_done == True:
            self.task_step = 0
            self.task_num += 1
            if self.task_num == len(self.data):
                is_done = True 
                self.task_num = 0 
        else:
            self.task_step += 1 

        return action.cpu().numpy()[0], base_action.cpu().numpy()[0], is_done, metrics

    # def update_critic(self, obs, action, base_next_action, reward, discount, next_obs, step):
    #     metrics = dict()

    #     if step % self.update_critic_every_steps != 0:
    #         return metrics

    #     with torch.no_grad():
    #         stddev = schedule(self.stddev_schedule, step)
    #         dist = self.actor(next_obs, base_next_action, stddev)

    #         offset_action = dist.sample(clip=self.stddev_clip)
    #         offset_action[:,:-7] *= self.hand_offset_scale_factor # NOTE: There is something wrong here?
    #         offset_action[:,-7:] *= self.arm_offset_scale_factor 
    #         next_action = base_next_action + offset_action

    #         target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
    #         target_V = torch.min(target_Q1, target_Q2)
    #         target_Q = reward + (discount * target_V)

    #     Q1, Q2 = self.critic(obs, action)

    #     critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

    #     # optimize encoder and critic
    #     self.critic_opt.zero_grad(set_to_none=True)
    #     critic_loss.backward()

    #     self.critic_opt.step()

    #     metrics['critic_target_q'] = target_Q.mean().item()
    #     metrics['critic_q1'] = Q1.mean().item()
    #     metrics['critic_q2'] = Q2.mean().item()
    #     metrics['critic_loss'] = critic_loss.item()
            
    #     return metrics

    # def update_actor(self, obs, base_action, step):
    #     metrics = dict()

    #     if step % self.update_actor_every_steps != 0:
    #         return metrics

    #     stddev = schedule(self.stddev_schedule, step)

    #     # compute action offset
    #     dist = self.actor(obs, base_action, stddev)
    #     action_offset = dist.sample(clip=self.stddev_clip)
    #     log_prob = dist.log_prob(action_offset).sum(-1, keepdim=True)

    #     # compute action
    #     # NOTE: Added for offset debug change
    #     # action_offset *= self.offset_mask
    #     # NOTE: Added for offset debug change
    #     action_offset[:,:-7] *= self.hand_offset_scale_factor
    #     action_offset[:,-7:] *= self.arm_offset_scale_factor 

    #     action = base_action + action_offset 
    #     Q1, Q2 = self.critic(obs, action)
    #     Q = torch.min(Q1, Q2)
    #     actor_loss = - Q.mean()

    #     # optimize actor
    #     self.actor_opt.zero_grad(set_to_none=True)
    #     actor_loss.backward()

    #     # NOTE: OFFSET MODE DEBUG
    #     # print('------')
    #     # total_norm = 0
    #     # for p in self.actor.parameters():
    #     #     param_norm = p.grad.data.norm(2)
    #     #     total_norm += param_norm.item() ** 2
    #     # total_norm = total_norm ** (1. / 2)
    #     # print('BEFORE CLIPPING TOTAL ACTOR GRADIENT NORM: {}'.format(total_norm))

    #     # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)

    #     # total_norm = 0
    #     # for p in self.actor.parameters():
    #     #     param_norm = p.grad.data.norm(2)
    #     #     total_norm += param_norm.item() ** 2
    #     # total_norm = total_norm ** (1. / 2)
    #     # print('AFTER CLIPPING ACTOR NORM: {}\n------'.format(total_norm))
    #     # NOTE: OFFSET MODE DEBUG 

    #     self.actor_opt.step()
        
    #     metrics['actor_loss'] = actor_loss.item()
    #     metrics['actor_logprob'] = log_prob.mean().item()
    #     metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
    #     metrics['actor_q'] = Q.mean().item()
    #     metrics['rl_loss'] = -Q.mean().item()

    #     return metrics
    
    def update(self, replay_iter, step):
        metrics = dict()

        if step % min(self.update_critic_frequency,
                      self.update_actor_frequency,
                      self.update_critic_target_frequency) != 0:
            return metrics

        batch = next(replay_iter)
        image_obs, tactile_repr, features, action, base_action, reward, discount, next_image_obs, next_tactile_repr, next_features, base_next_action = to_torch(
            batch, self.device)
        
        # Multiply action with the offset mask just incase if the buffer was not saved that way
        offset_action = action - base_action
        offset_action *= self.offset_mask[self.task_num]
        action = base_action + offset_action

        # Get the representations
        obs = self._get_policy_reprs_from_obs(
            image_obs = image_obs,
            tactile_repr = tactile_repr,
            features = features,
            representation_types=self.policy_representations,
            task_num = self.task_num
        )

        obs_aug = None
        if self.seprarte_aug:
            obs_aug = self._get_policy_reprs_from_obs(
                image_obs = image_obs,
                tactile_repr = tactile_repr,
                features = features, 
                representation_type=self.policy_representations,
                task_num = self.task_num
            )

        next_obs = self._get_policy_reprs_from_obs(
            image_obs = next_image_obs, 
            tactile_repr = next_tactile_repr,
            features = next_features,
            representation_types=self.policy_representations,
            task_num = self.task_num 
        )

        next_obs_aug = None
        if self.separate_aug:
            next_obs_aug = self._get_policy_reprs_from_obs(
                image_obs = next_image_obs,
                tactile_repr = next_tactile_repr,
                features = next_features,
                representation_types=self.policy_representations,
                task_num = self.task_num
            ) 


        metrics['batch_reward'] = reward.mean().item()

        # # update critic
        # metrics.update(
        #     self.update_critic(obs, action, base_next_action, reward, discount, next_obs, step))

        # # update actor
        # metrics.update(
        #     self.update_actor(obs.detach(), base_action, step))

        # # update critic target
        # soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        if step % self.update_critic_frequency == 0:
            metrics.update(
                self.rl_learner.update_critic(
                    obs=obs,
                    obs_aug=obs_aug,
                    action=action,
                    base_next_action=base_next_action,
                    reward=reward,
                    next_obs=next_obs,
                    next_obs_aug=next_obs_aug,
                    discount=discount,
                    step=step
                )
            )

        # with torch.autograd.set_detect_anomaly(True):
        if step % self.update_actor_frequency == 0:
            metrics.update(
                self.rl_learner.update_actor(
                    obs=obs,
                    base_action=base_action,
                    step=step
                )
            )

        if step % self.update_critic_target_frequency == 0:
            self.rl_learner.update_critic_target()

        return metrics

    def get_reward(self, episode_obs, episode_id, visualize=False): # TODO: Delete the mock option
        
        final_reward, final_cost_matrix, best_expert_id = self.rewarder[self.task_num].get(
            obs = episode_obs
        )

        # Update the reward scale if it's the first episode
        if episode_id == 1:
            # Auto update the reward scale and get the rewards again
            self.rewarder[self.task_num].update_scale(current_rewards = final_reward)
            final_reward, final_cost_matrix, best_expert_id = self.rewarder[self.task_num].get(
                obs = episode_obs
            )

        print('final_reward: {}, best_expert_id: {}'.format(final_reward, best_expert_id))

        if visualize:
            self.plot_cost_matrix(final_cost_matrix, expert_id=best_expert_id, episode_id=episode_id)

        return final_reward
    
    def plot_cost_matrix(self, cost_matrix, expert_id, episode_id, file_name=None):
        if file_name is None:
            ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            file_name = f'{ts}_expert_{expert_id}_ep_{episode_id}_cost_matrix.png'

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

    # def save_snapshot(self):
    #     keys_to_save = ['actor', 'critic', 'image_encoder']
    #     payload = {k: self.__dict__[k] for k in keys_to_save}
    #     return payload

    # def load_snapshot(self, payload):
    #     loaded_encoder = None
    #     for k, v in payload.items():
    #         self.__dict__[k] = v
    #         if k == 'image_encoder':
    #             loaded_encoder = v

    #     self.critic_target.load_state_dict(self.critic.state_dict())
    #     self.image_encoder[self.task_num].load_state_dict(loaded_encoder.state_dict()) # NOTE: In the actual repo they use self.vinn_encoder rather than loaded_encoder 

    #     self.image_encoder[self.task_num].eval()
    #     for param in self.image_encoder[self.task_num].parameters():
    #         param.requires_grad = False

    #     self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
    #     self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    # def load_snapshot_eval(self, payload, bc=False):
    #     for k, v in payload.items():
    #         self.__dict__[k] = v
    #     self.critic_target.load_state_dict(self.critic.state_dict()) 

    #     self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
    #     self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def save_snapshot(self):
        return self.rl_learner.save_snapshot()
    
    def load_snapshot(self, payload):
        return self.rl_learner.load_snapshot(payload)
    
    def load_snapshot_eval(self, payload):
        return self.rl_learner.load_snapshot_eval(payload)
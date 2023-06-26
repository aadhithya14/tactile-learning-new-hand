# Distil agent, here we don't really have a rewarder but we will be using the mse loss between the
# expert agents and our current agent

import random 
import hydra
import torch 

from tactile_learning.utils import TruncatedNormal, schedule, to_torch
from tactile_learning.models import MLP

from torch import nn 
from torch.nn import functional as F

from .agent import Agent
# from .base_policy.vinn import VINN

class Actor(nn.Module):
    def __init__(self, repr_dim, goal_dim, action_shape, hidden_dim):
        super().__init__()

        self._output_dim = action_shape[0]
        # if repr_dim != 512:
        #     goal_dim = repr_dim - 1 if repr_dim%2==1 else repr_dim
        # else:
        #     goal_dim = repr_dim
        
        self.policy = nn.Sequential(nn.Linear(repr_dim + goal_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True))
        
        self._head = MLP(in_channels=hidden_dim, hidden_channels=[self._output_dim])

    def forward(self, obs, goal, std):

        obs = torch.cat([goal, obs], dim=-1)
        feat = self.policy(obs)

        mu = torch.tanh(self._head(feat))
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)
        return dist
    
# This agent will receive multiple expert offset policies in load_snapshot
# It also loads multiple base policies
# Then at each time step it finds the closest neighbor
# Uses the goal of that neighbor to condition the final policy
# Receives the nearest neighbor of the state, gets the base action and calculates the offset action to find the expert policy
# Then uss that action to get the mse loss
class NeighborDistilAgent(Agent): # NOTE: For now let's try distilling randomly
    def __init__(
        self,
        data_path, # Agent parameters
        expert_demo_nums,
        image_out_dir, image_model_type,
        tactile_out_dir, tactile_model_type,
        view_num, 
        device,
        lr, update_every_steps, stddev_schedule, stddev_clip, features_repeat,
        hidden_dim, policy_representations, goal_representations, # Distil agent 
        hand_offset_scale_factor, arm_offset_scale_factor,
        **kwargs
    ):
        super().__init__(
            data_path=data_path,
            expert_demo_nums=expert_demo_nums,
            image_out_dir=image_out_dir, image_model_type=image_model_type,
            tactile_out_dir=tactile_out_dir, tactile_model_type=tactile_model_type,
            view_nun=view_num, device=device, lr=lr, update_every_steps=update_every_steps,
            stddev_schedule=stddev_schedule, stddev_clip=stddev_clip,
            features_repeat=features_repeat, **kwargs)
        
        # Offset scale factors
        self.hand_offset_scale_factor = hand_offset_scale_factor 
        self.arm_offset_scale_factor = arm_offset_scale_factor

        # Models - encoders are already set in the main agent
        self.goal_representations = goal_representations
        self.policy_representations = policy_representations
        self.actor = Actor(
            repr_dim=self.repr_dim('policy'),
            goal_dim=self.repr_dim('goal'),
            action_shape=23, hidden_dim=hidden_dim).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def __repr__(self):
        return "distil_agent"

    def train(self, training=True):
        self.training = training
        if training:
            self.actor.train(training)
        else:
            self.actor.eval()

    def initialize_modules(self, vinn_cfg): 
        self.vinn = hydra.utils.instatiate(
            vinn_cfg,
            expert_demos = self.expert_demos, 
            tactile_repr_size = self.tactile_repr.size
        )

    # Act should find the nearest neighbor and return it's ID so we should receive the neighbor ID from the buffer
    def act(self, obs, global_step, episode_step, eval_mode, metrics=None):
        # Find the ID of the nearest neighbor demo
        _, _, neighbor_demo_id = self.vinn.act(obs=obs, episode_step=episode_step, get_id=True) 

        # Get the nearest neighbor goal as a policy representation
        neighbor_goal_img = self.expert_demos[neighbor_demo_id]['image_obs'][-1]
        neighbor_goal_tactile_repr = self.expert_demos[neighbor_demo_id]['tactile_repr'][-1]
        with torch.no_grad():
            neighbor_goal = self._get_policy_reprs_from_obs(
                image_obs = neighbor_goal_img, 
                tactile_repr = neighbor_goal_tactile_repr,
                features = None, 
                representation_types = self.goal_representations
            )

        # Get the current policy observation
        with torch.no_grad():
            # Get the action image_obs
            obs = self._get_policy_reprs_from_obs( # This method is called with torch.no_grad() in training anyways
                image_obs = obs['image_obs'].unsqueeze(0) / 255.,
                tactile_repr = obs['tactile_repr'].unsqueeze(0),
                features = obs['features'].unsqueeze(0),
                representation_types=self.policy_representations
            )

        # Get the action from the actor
        stddev = schedule(self.stddev_schedule, global_step)
        dist = self.actor(obs, neighbor_goal, stddev)

        if eval_mode:
            action = dist.mean 
        else:
            action = dist.sample(clip=None)

        return action.cpu().numpy()[0], neighbor_goal, neighbor_demo_id # For the specific policies these ids will be the 
    
    def update(self, replay_iter, global_step):
        metrics = dict()

        # NOTE: We're saving goal directly for now - other alternatives could be thought
        # We should always save the expert ID for sure, but also goal could make things faster
        # When we're training we can choose the expert to learn randomly
        env_idx = random.randint(0, len(replay_iter)-1)
        batch = next(replay_iter[env_idx]) # We'll be getting that expert's only observation and the goal
        image_obs, tactile_repr, features, base_action, goal = to_torch(batch, self.device) # TODO: Should rewrite the replay buffer
        
        with torch.no_grad():
            obs = self._get_policy_reprs_from_obs(
                image_obs = image_obs, 
                tactile_repr = tactile_repr,
                features = features,
                representation_types=self.policy_representations
            ) # NOTE: We're freezing the encoders for now

        # Get the current predicted action
        stddev = schedule(self.stddev_schedule, global_step)
        dist = self.actor(obs, goal, stddev)
        curr_action = dist.mean

        # Get the predicted offset action of the neighbor
        dist_offset_neighbor = self.actor_expert[env_idx](obs, base_action, stddev) # Should calculate the offset action for all
        offset_neighbor_action = dist_offset_neighbor.mean
        offset_neighbor_action[:,:-7] *= self.hand_offset_scale_factor
        offset_neighbor_action[:,-7:] *= self.arm_offset_scale_factor 
        neighbor_action = base_action + offset_neighbor_action

        # Calculate the loss
        actor_loss = F.mse_loss(curr_action, neighbor_action, reduction='mean')

        # Update 
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()

        return metrics
        
    def repr_dim(self, type='policy'):
        representations = self.policy_representations if type=='policy' else self.goal_representations
        repr_dim = 0
        if 'tactile' in representations:
            repr_dim += self.tactile_repr.size
        if 'image' in representations:
            repr_dim += 512
        if 'features' in representations:
            repr_dim += 23 * self.features_repeat

        return repr_dim
    
    def save_snapshot(self):
        super().save_snapshot(
            keys_to_save=['actor']
        )
    
    def load_snapshot(self, payload, env_idx):
        if env_idx == 0: # If we're loading the first environment
            self.actor_expert = []
        
        for k, v in payload.items():
            if k == 'actor':
                self.actor_expert.append(v)

        # Turn off the graduents for the experts
        for param in self.actor_expert[env_idx].parameters():
            param.requires_grad = False

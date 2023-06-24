# Distil agent, here we don't really have a rewarder but we will be using the mse loss between the
# expert agents and our current agent

import random 
import torch 

from tactile_learning.utils import TruncatedNormal
from tactile_learning.models import MLP

from torch import nn 
from torch.nn import functional as F

from .agent import Agent

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        self._output_dim = action_shape[0]
        if repr_dim != 512:
            goal_dim = repr_dim - 1 if repr_dim%2==1 else repr_dim
        else:
            goal_dim = repr_dim
        
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
class NeighborDistilAgent(Agent):
    def __init__(
        self,
        data_path,
        expert_demo_nums,
        image_out_dir, image_model_type,
        tactile_out_dir, tactile_model_type,
        view_num, 
        device,
        features_repeat,
        policy_representations,
        experiment_name,
        lr,
        
    ):
        
        
    @property
    def repr_dim(self):
        repr_dim = 0
        if 'tactile' in self.policy_representations:
            repr_dim += self.tactile_repr.size
        if 'image' in self.policy_representations:
            repr_dim += 512
        if 'features' in self.policy_representations:
            repr_dim += 23 * self.features_repeat

        return repr_dim
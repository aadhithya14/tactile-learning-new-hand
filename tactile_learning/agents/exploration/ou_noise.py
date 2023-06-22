import numpy as np
import torch

from .explorer import Explorer
from tactile_learning.utils import OrnsteinUhlenbeckActionNoise

class OUNoise(Explorer):
    def __init__(
        self, 
        num_expl_steps,
        sigma=0.8
    ):
        super().__init__(num_expl_steps=num_expl_steps)
        self.ou_noise = OrnsteinUhlenbeckActionNoise(
            mu = np.zeros(23), # The mean of the offsets should be 0
            sigma = sigma # It will give bw -1 and 1 - then this gets multiplied by the scale factors ...
        )

    def explore(self, offset_action, global_step, device):
        if global_step < self.num_expl_steps:
            offset_action = torch.FloatTensor(self.ou_noise()).to(device).unsqueeze(0)

        return offset_action
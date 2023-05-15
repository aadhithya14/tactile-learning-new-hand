# Script for implementing custom encoders

import torch.nn as nn
# import utils

from tactile_learning.models import weight_init

# Module to print out the shape of the conv layer - used to debug
class PrintSize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x


class ImageBCEncoder(nn.Module):
    def __init__(self, out_dim, in_channels):
        super().__init__()

        # assert len(in_channels) == 3

        self.convnet = nn.Sequential(nn.Conv2d(in_channels, 32, 6, stride=4),
                                     nn.ReLU(), nn.Conv2d(32, 32, 6, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 6, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 6, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 4, stride=1),
                                     nn.ReLU())

        self.trunk = nn.Sequential(nn.Linear(10368, out_dim),
								   nn.LayerNorm(out_dim), nn.Tanh())

        # self.apply(weight_init) - For some reason this is giving an error

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.contiguous().view(h.shape[0], -1)
        # print('h.shape in forward: {}'.format(h.shape))
        h = self.trunk(h)
        return h
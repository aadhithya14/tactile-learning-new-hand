# Given a buffer root - modify the reward at each timestep
# at that buffer with the given rewards

import os
import hydra
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image
from torchvision import transforms as T
from PIL import Image
# from agent.encoder import Encoder

from tactile_learning.models import *
from tactile_learning.utils import *
from tactile_learning.tactile_data import *
from tactile_learning.datasets import *

fn = '/home/irmak/Workspace/tactile-learning/buffer/2023.05.22T18-17_training_episodes/20230523T120224_80_76.npz'

episode = load_episode(fn) 

print(episode['rewards'])
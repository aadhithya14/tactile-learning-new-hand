# Script that turns on an offset with a set schedule
# There will be an exponential time difference between each offset unlock
# It will receive an order to unlock at each unlocking phase
# The policy should explore for longer and longer with each unlocking phase

import numpy as np 
import torch 

from .explorer import Explorer
from tactile_learning.utils import OrnsteinUhlenbeckActionNoise

class ScheduledGuidance(Explorer):
    def __init__(
        self,
        
    )
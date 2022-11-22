import hydra 
from torch.nn.parallel import DistributedDataParallel as DDP

import tactile_learning.models.custom import TactileJointLinear
import tactile_learning.models.agents.behavior_cloning import TactileJointBC

def init_bc(cfg, device, rank): 
    model = TactileJointLinear()
# Helper script to load models
import numpy as np
import os
import torch
import torch.utils.data as data 

from collections import OrderedDict
from tqdm import tqdm 
from torch.nn.parallel import DistributedDataParallel as DDP

from tactile_learning.models.custom import TactileJointLinear

def load_model(cfg, device, model_path):
    if cfg.agent_type == 'bc':
        model = TactileJointLinear(input_dim=cfg.tactile_info_dim + cfg.joint_pos_dim,
                                   output_dim=cfg.joint_pos_dim,
                                   hidden_dim=cfg.hidden_dim)
    state_dict = torch.load(model_path)
    
    # Modify the state dict accordingly - this is needed when multi GPU saving was done
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v 
    
    # Load the new state dict to the model 
    model.load_state_dict(new_state_dict)

    # Turn it into DDP - it was saved that way 
    model = DDP(model.to(device), device_ids=[0])

    return model 
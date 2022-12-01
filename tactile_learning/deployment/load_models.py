# Helper script to load models
import numpy as np
import os
import torch
import torch.utils.data as data 

from collections import OrderedDict
from tqdm import tqdm 
from torch.nn.parallel import DistributedDataParallel as DDP

from tactile_learning.models.custom import TactileJointLinear, TactileImageEncoder

def load_model(cfg, device, model_path):
    # Initialize the model
    if cfg.agent_type == 'bc':
        model = TactileJointLinear(
            input_dim=cfg.tactile_info_dim + cfg.joint_pos_dim,
            output_dim=cfg.joint_pos_dim,
            hidden_dim=cfg.hidden_dim
        )
    elif cfg.agent_type == 'byol': # load the encoder
        model = TactileImageEncoder(
            in_channels=cfg.encoder.in_channels,
            out_dim=cfg.encoder.out_dim
        )
    # print('model: {}'.format(model))
    state_dict = torch.load(model_path)
    
    # Modify the state dict accordingly - this is needed when multi GPU saving was done
    new_state_dict = modify_multi_gpu_state_dict(state_dict)
    
    if cfg.agent_type == 'byol':
        new_state_dict = modify_byol_state_dict(new_state_dict)

    # Load the new state dict to the model 
    model.load_state_dict(new_state_dict)

    # Turn it into DDP - it was saved that way 
    model = DDP(model.to(device), device_ids=[0])

    return model

def modify_multi_gpu_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v 
    return new_state_dict

def modify_byol_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'encoder.net' in k:
            name = k[12:] # Everything after encoder.net
            new_state_dict[name] = v
    # print(new_state_dict['encoder.net'])
    return new_state_dict

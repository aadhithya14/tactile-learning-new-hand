# Helper script to load models
import hydra
import numpy as np
import torch
import torch.utils.data as data 

from collections import OrderedDict
from tqdm import tqdm 
from torch.nn.parallel import DistributedDataParallel as DDP

from tactile_learning.models.custom import *

def load_model(cfg, device, model_path, bc_model_type=None):
    # Initialize the model
    print('cfg.learner_type: {}'.format(cfg.learner_type))
    if cfg.learner_type == 'bc':
        if bc_model_type == 'image':
            model = hydra.utils.instantiate(cfg.encoder.image_encoder)
        elif bc_model_type == 'tactile':
            model = hydra.utils.instantiate(cfg.encoder.tactile_encoder)
        elif bc_model_type == 'last_layer':
            model = hydra.utils.instantiate(cfg.encoder.last_layer)

    elif 'byol' in cfg.learner_type: # load the encoder
        model = hydra.utils.instantiate(cfg.encoder) # NOTE: Wouldm't this work? 

    state_dict = torch.load(model_path)
    
    # Modify the state dict accordingly - this is needed when multi GPU saving was done
    new_state_dict = modify_multi_gpu_state_dict(state_dict)
    
    if 'byol' in cfg.learner_type:
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

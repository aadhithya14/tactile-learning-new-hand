import os
import hydra
import torch
# import torch.nn as nn

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Get the byol wrapper
from tactile_learning.models.ssl_wrappers.byol import BYOL
from tactile_learning.utils.augmentations import get_tactile_augmentations
from tactile_learning.utils.constants import *

# Agent to train an encoder with byol

class TactileImageBYOL:
    def __init__(
        self,
        byol,
        optimizer
    ):

        self.optimizer = optimizer 
        self.byol = byol

    def to(self, device):
        self.device = device 
        self.byol.to(device)

    def train(self):
        self.byol.train()

    def eval(self):
        self.byol.eval()

    def save(self, checkpoint_dir):
        torch.save(self.byol.state_dict(),
                   os.path.join(checkpoint_dir, 'byol_encoder.pt'),
                   _use_new_zipfile_serialization=False)

    def train_epoch(self, train_loader):
        self.train() 

        # Save the train loss
        train_loss = 0.0 

        # Training loop 
        for batch in train_loader: 
            tactile_image, _, _ = [b.to(self.device) for b in batch]
            self.optimizer.zero_grad()

            # print('tactile_image.shape: {}'.format(tactile_image.shape))

            # Get the loss by the byol            
            loss = self.byol(tactile_image)
            train_loss += loss.item() 

            # Backprop
            loss.backward() 
            self.optimizer.step()
            self.byol.update_moving_average() 

        return train_loss / len(train_loader)

    

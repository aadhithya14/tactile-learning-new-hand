import os
import hydra
import torch
# import torch.nn as nn

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm 

# Custom imports 
from tactile_learning.utils.losses import mse, l1

# Agent to get current state and predict the action applied
# It will learn in supervised way
# 
class TactileJointBC:
    def __init__(self,
                 model,
                 optimizer,
                 loss_fn: str) -> None:
        
        self.model = model # Model will be bcdist or bcregular according to what we want
        self.optimizer = optimizer 

        if loss_fn == 'mse':
            self.loss_fn = mse 
        elif loss_fn == 'l1':
            self.loss_fn = l1 

    def to(self, device):
        self.device = device
        self.model.to(device)

    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

    def save(self, checkpoint_dir):
        torch.save(self.model.state_dict(),
                   os.path.join(checkpoint_dir, 'bc_model_pd_{}.pt'.format(self.predict_dist)),
                   _use_new_zipfile_serialization=False)

    def train_epoch(self, train_loader):
        self.train() # Set the agent to training mode

        # Save the train loss 
        train_loss = 0.0

        # Training loop
        for batch in train_loader:
            import ipdb; ipdb.set_trace() 
            self.optimizer.zero_grad()
            _, tactile_info, joint_pos, action = [b.to(self.device)for b in batch]
            bs = tactile_info.shape[0]
            flattened_tactile = torch.reshape(tactile_info, (bs,-1))
            all_info = torch.cat((flattened_tactile, joint_pos), -1) # Concatenate at the last axis
            
            pred_action = self.model(all_info)

            # Get the supervised loss
            loss = self.loss_fn(action, pred_action)
            train_loss += loss.item()

            # Backprop
            loss.backward()
            self.optimizer.step()

        return train_loss / len(train_loader) # Average loss for all values in a batch 

    def test_epoch(self, test_loader):
        self.eval() 

        # Save the test loss
        test_loss = 0.0

        # Test for one epoch
        for batch in test_loader:
            _, tactile_info, joint_pos, action = [b.to(self.device) for b in batch]
            with torch.no_grad():
                flattened_tactile = torch.reshape(tactile_info, (bs,-1))
                all_info = torch.cat((flattened_tactile, joint_pos), -1) # Concatenate at the last axis
                pred_action = self.model(all_info)
            
            loss = self.loss_fn(action, pred_action)
            test_loss += loss.item()

        return test_loss / len(test_loader)
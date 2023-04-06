# Learner script for gaussian mixture model on top of the behavior cloning 

import os
import torch

from torch.distributions import Categorical, Normal

from .learner import Learner

class BCGMM(Learner):
    def __init__(
        self,
        image_encoder,
        tactile_encoder,
        last_layer,
        optimizer,
        representation_type='t-dex',  # We will use image and tactile images for now 
        freeze_encoders=True # We will try both of them
    ):
        self.image_encoder = image_encoder 
        self.tactile_encoder = tactile_encoder
        self.last_layer = last_layer 
        self.optimizer = optimizer 
        self.representation_type = representation_type
        self.freeze_encoders = freeze_encoders

    def to(self, device): 
        self.device = device
        self.image_encoder.to(device)
        self.tactile_encoder.to(device)
        self.last_layer.to(device)

    def train(self):
        self.last_layer.train() 
        if not self.freeze_encoders:
            self.image_encoder.train() 
            self.tactile_encoder.train()

    def eval(self):
        self.image_encoder.eval()
        self.tactile_encoder.eval()
        self.last_layer.eval()

    def save(self, checkpoint_dir, model_type='best'):
        torch.save(self.image_encoder.state_dict(),
                   os.path.join(checkpoint_dir, f'bc_gmm_image_encoder_{model_type}.pt'),
                   _use_new_zipfile_serialization=False)

        torch.save(self.tactile_encoder.state_dict(),
                   os.path.join(checkpoint_dir, f'bc_gmm_tactile_encoder_{model_type}.pt'),
                   _use_new_zipfile_serialization=False)

        torch.save(self.last_layer.state_dict(),
                   os.path.join(checkpoint_dir, f'bc_gmm_last_layer_{model_type}.pt'),
                   _use_new_zipfile_serialization=False)

    def _get_all_repr(self, tactile_image, vision_image):
        if self.freeze_encoders:
            with torch.no_grad():
                tactile_repr = self.tactile_encoder(tactile_image)
                vision_repr = self.image_encoder(vision_image)
        else:
            tactile_repr = self.tactile_encoder(tactile_image)
            vision_repr = self.image_encoder(vision_image)
        
        if self.representation_type == 't-dex':
            all_repr = torch.concat((tactile_repr, vision_repr), dim=-1)
            return all_repr
        if self.representation_type == 'tactile':
            return tactile_repr 
        if self.representation_type == 'image':
            return vision_repr

    def _get_gmm_loss(self, all_repr, target_actions): # all_repr.shape: (B,repr_shape), target_actions: (B,A)
        logits, mu, sigma = self.last_layer(all_repr) # (BatchSize, NumofGuassians, ActionSpace) (B,G,A)
        gaussians = Normal(mu, sigma) # B,G,A
        pi = torch.softmax(logits, dim=1) # B,G - should be 
        logprob = gaussians.log_prob(target_actions.unsqueeze(1)).sum(-1) # Shape: (B,G) Finds the each probability of the gaussian clusters
        prob = pi * torch.exp(logprob) # B,G

        nll = -torch.log(prob.sum(dim=-1)) # B
        loss = nll.mean()

        return loss

    def train_epoch(self, train_loader):
        self.train() 

        train_loss = 0.

        for batch in train_loader:
            self.optimizer.zero_grad() 
            tactile_image, vision_image, action = [b.to(self.device) for b in batch]
            all_repr = self._get_all_repr(tactile_image, vision_image)

            loss = self._get_gmm_loss(all_repr, action)
            train_loss += loss.item()

            loss.backward() 
            self.optimizer.step()

        return train_loss / len(train_loader)

    def test_epoch(self, test_loader):
        self.eval() 

        test_loss = 0.

        for batch in test_loader:
            tactile_image, vision_image, action = [b.to(self.device) for b in batch]
            with torch.no_grad():        
                all_repr = self._get_all_repr(tactile_image, vision_image)
                loss = self._get_gmm_loss(all_repr, action) 

            test_loss += loss.item()

        return test_loss / len(test_loader)
import os
import torch

from .learner import Learner 

class SIMCLRLearner(Learner):
    def __init__(
        self,
        simclr_wrapper,
        optimizer
    ):

        self.optimizer = optimizer 
        self.wrapper = simclr_wrapper
        
    def to(self, device):
        self.device = device 
        self.wrapper.to(device)

    def train(self):
        self.wrapper.train()

    def eval(self):
        self.wrapper.eval()

    def save(self, checkpoint_dir, model_type='best'):
        encoder_weights = self.wrapper.get_encnoder_weights()
        torch.save(
            encoder_weights,
            os.path.join(checkpoint_dir, f'ssl_encoder_{model_type}.pt'),
            _use_new_zipfile_serialization=False
        )

    def train_epoch(self, train_loader):
        self.train() 

        # Save the train loss
        train_loss = 0.0 

        # Training loop 
        for batch in train_loader: 
            image = batch.to(self.device)
            self.optimizer.zero_grad()

            # Get the loss by the byol            
            loss = self.wrapper.forward(image)
            train_loss += loss.item() 

            # Backprop
            loss.backward() 
            self.optimizer.step()

        return train_loss / len(train_loader)
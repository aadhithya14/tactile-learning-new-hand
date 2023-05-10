import os
import torch

from .learner import Learner 
from tactile_learning.models import adjust_moco_momentum

class MOCOLearner(Learner):
    def __init__(
        self,
        wrapper,
        optimizer,
        momentum, 
        total_epochs
    ):

        self.epoch_num = 0
        self.optimizer = optimizer 
        self.wrapper = wrapper
        self.momentum = momentum
        self.total_epochs = total_epochs
        
    def to(self, device):
        self.device = device 
        self.wrapper.to(device)

    def train(self):
        self.wrapper.train()

    def eval(self):
        self.wrapper.eval()

    def save(self, checkpoint_dir, model_type='best'):
        encoder_weights = self.wrapper.get_encoder_weights()
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
        for idx, batch in enumerate(train_loader): 
            image = batch.to(self.device)
            self.optimizer.zero_grad()

            # Get the loss by the byol   
            moco_momentum = adjust_moco_momentum(
                epoch = self.epoch_num + (idx / len(train_loader)), 
                momentum = self.momentum,
                total_epochs = self.total_epochs
            )
            loss = self.wrapper.forward(image, moco_momentum)
            train_loss += loss.item() 

            # Backprop
            loss.backward() 
            self.optimizer.step()

        self.epoch_num += 1

        return train_loss / len(train_loader)
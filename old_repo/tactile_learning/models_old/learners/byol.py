import os
import torch

class BYOLLearner:
    def __init__(
        self,
        byol,
        optimizer,
        byol_type
    ):

        self.optimizer = optimizer 
        self.byol = byol
        self.byol_type = byol_type # Tactile or Image

    def to(self, device):
        self.device = device 
        self.byol.to(device)

    def train(self):
        self.byol.train()

    def eval(self):
        self.byol.eval()

    def save(self, checkpoint_dir, model_type='best'):
        torch.save(self.byol.state_dict(),
                   os.path.join(checkpoint_dir, f'byol_encoder_{model_type}.pt'),
                   _use_new_zipfile_serialization=False)

    def train_epoch(self, train_loader):
        self.train() 

        # Save the train loss
        train_loss = 0.0 

        # print('len(dataloader): {}'.format(len(train_loader)))

        # Training loop 
        for batch in train_loader: 
            # if self.byol_type == 'tactile':
            #     # image, _ = [b.to(self.device) for b in batch] # NOTE: Be careful here
            #     image = batch.to(self.device) # NOTE: assumes that it uses single tactile giving dataset
            # elif self.byol_type == 'image': # TODO: For BYOL we actually should always use dataset that gives one output only
            #     _, image = [b.to(self.device) for b in batch]
            image = batch.to(self.device)
            self.optimizer.zero_grad()

            # print('image.shape: {}'.format(image.shape))

            # Get the loss by the byol            
            loss = self.byol(image)
            train_loss += loss.item() 

            # Backprop
            loss.backward() 
            self.optimizer.step()
            self.byol.update_moving_average() 

        return train_loss / len(train_loader)

    
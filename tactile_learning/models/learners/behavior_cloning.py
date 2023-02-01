import os
import hydra
import torch
# import torch.nn as nn

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm 

# Custom imports 
from tactile_learning.utils.losses import mse, l1

# Learner to get current state and predict the action applied
# It will learn in supervised way
# 
class ImageTactileBC:
    # Model that takes in two encoders one for image, one for tactile and puts another linear layer on top
    # then, with each image and tactile image it passes through them through the encoders, concats the representations
    # and passes them through another linear layer and gets the actions
    def __init__(
        self,
        image_encoder,
        tactile_encoder, 
        last_layer,
        optimizer,
        loss_fn,
        representation_type, # image, tactile, all
    ):

        self.image_encoder = image_encoder 
        self.tactile_encoder = tactile_encoder
        self.last_layer = last_layer  
        self.optimizer = optimizer 
        self.representation_type = representation_type

        if loss_fn == 'mse':
            self.loss_fn = mse
        elif loss_fn == 'l1':
            self.loss_fn = l1

    def to(self, device):
        self.device = device
        self.image_encoder.to(device)
        self.tactile_encoder.to(device)
        self.last_layer.to(device)

    def train(self):
        self.image_encoder.train()
        self.tactile_encoder.train()
        self.last_layer.train()
    
    def eval(self):
        self.image_encoder.eval()
        self.tactile_encoder.eval()
        self.last_layer.eval()

    def save(self, checkpoint_dir, model_type='best'):
        torch.save(self.image_encoder.state_dict(),
                   os.path.join(checkpoint_dir, f'bc_image_encoder_{model_type}.pt'),
                   _use_new_zipfile_serialization=False)

        torch.save(self.tactile_encoder.state_dict(),
                   os.path.join(checkpoint_dir, f'bc_tactile_encoder_{model_type}.pt'),
                   _use_new_zipfile_serialization=False)

        torch.save(self.last_layer.state_dict(),
                   os.path.join(checkpoint_dir, f'bc_last_layer_{model_type}.pt'),
                   _use_new_zipfile_serialization=False)

    def _get_all_repr(self, tactile_image, vision_image):
        if self.representation_type == 'all':
            tactile_repr = self.tactile_encoder(tactile_image)
            vision_repr = self.image_encoder(vision_image)
            all_repr = torch.concat((tactile_repr, vision_repr), dim=-1)
            return all_repr
        if self.representation_type == 'tactile':
            tactile_repr = self.tactile_encoder(vision_image)
            return tactile_repr 
        if self.representation_type == 'image':
            vision_repr = self.image_encoder(vision_image)
            return vision_repr


    def train_epoch(self, train_loader):
        self.train() 

        train_loss = 0.

        for batch in train_loader:
            self.optimizer.zero_grad() 
            tactile_image, vision_image, action = [b.to(self.device) for b in batch]
            all_repr = self._get_all_repr(tactile_image, vision_image)
            pred_action = self.last_layer(all_repr)

            loss = self.loss_fn(action, pred_action)
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
                tactile_repr = self.tactile_encoder(tactile_image)
                vision_repr = self.image_encoder(vision_image)
                all_repr = torch.concat((tactile_repr, vision_repr), dim=-1)
                pred_action = self.last_layer(all_repr)

            loss = self.loss_fn(action, pred_action)
            test_loss += loss.item()

        return test_loss / len(test_loader)

from tactile_learning.models.pretrained import resnet18
from tactile_learning.models.custom import TactileStackedEncoder
from tactile_learning.models.utils import create_fc
from tactile_learning.datasets.all import TactileVisionActionDataset 
from torch.utils import data

if __name__ == '__main__':
    image_encoder = resnet18(pretrained = True) # output will be 512
    tactile_encoder = TactileStackedEncoder(45, 512)
    last_layer = create_fc(1024, 23, [512, 128, 64]) # 23 - allegro_action + kinova_action
    optim_params = list(image_encoder.parameters()) + list(tactile_encoder.parameters()) + list(last_layer.parameters())
    optimizer = torch.optim.Adam(params=optim_params, 
                                 lr = 1e-3, 
                                 weight_decay=1e-5)
    learner = ImageTactileBC(
        image_encoder,
        tactile_encoder, 
        last_layer,
        optimizer,
        loss_fn = 'mse'
    )

    dset = TactileVisionActionDataset(
        data_path = '/home/irmak/Workspace/Holo-Bot/extracted_data/cup_slipping/eval',
        tactile_information_type = 'stacked',
        tactile_img_size=16,
        vision_view_num=1
    ) 
    dataloader = data.DataLoader(dset, 
                                batch_size  = 128, 
                                shuffle     = True, 
                                num_workers = 8,
                                pin_memory  = True)

    train_loss = learner.train_epoch(dataloader)

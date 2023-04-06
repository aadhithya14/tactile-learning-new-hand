# Helper script to load models
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image as im
from omegaconf import OmegaConf
from tqdm import tqdm 
from torch.distributions import Categorical, Normal
from torchvision.datasets.folder import default_loader as loader
from torchvision import models

from holobot.constants import *
from holobot.utils.network import ZMQCameraSubscriber
from holobot.robot.allegro.allegro_kdl import AllegroKDL

from torchvision.transforms.functional import crop

from tactile_learning.models import load_model, init_encoder_info
from tactile_learning.utils import *
from tactile_learning.tactile_data import *

from .deployer import Deployer

class BCGMM(Deployer):
    def __init__(
        self,
        data_path,
        deployment_dump_dir,
        out_dir, # We will be experimenting with the trained encoders with bc
        view_num = 1,
        representation_type = 'tdex'
    ):
        self.set_up_env()

        self.device = torch.device('cuda:0')
        self.view_num = view_num
        self.representation_type = representation_type
        self.out_dir = out_dir

        self._initalize_models()

        self.state_id = 0
        self.deployment_dump_dir = deployment_dump_dir
        os.makedirs(self.deployment_dump_dir, exist_ok=True)

    def _initalize_models(self):
        # Get the BC_GMM training config
        cfg = OmegaConf.load(os.path.join(self.out_dir, '.hydra/config.yaml'))

        self.image_cfg, self.image_encoder, self.image_transform = init_encoder_info(
            device = self.device,
            out_dir = cfg.learner.image_out_dir,
            encoder_type = 'image',
            view_num = 1
        )
        self.inv_image_transform = get_inverse_image_norm()

        tactile_cfg, tactile_encoder, _ = init_encoder_info(
            device = self.device,
            out_dir = cfg.learner.tactile_out_dir,
            encoder_type = 'tactile'
        )
        # print('tactile_encoder.device: {}'.format(tactile_encoder))
        self.tactile_img = TactileImage(
            tactile_image_size = tactile_cfg.tactile_image_size, 
            shuffle_type = None
        )
        self.tactile_repr = TactileRepresentation(
            encoder_out_dim = tactile_cfg.encoder.out_dim,
            tactile_encoder = tactile_encoder,
            tactile_image = self.tactile_img,
            representation_type = self.representation_type
        )

        self.gmm_layer = load_model(
            cfg = cfg, 
            device = self.device,
            model_path = os.path.join(self.out_dir, 'models/bc_gmm_last_layer_best.pt')
        )

    def _get_all_repr(self, tactile_values, vision_image):
        tactile_repr = torch.FloatTensor(self.tactile_repr.get(tactile_values)).to(self.device).unsqueeze(0)

        vision_repr = self.image_encoder(vision_image.unsqueeze(dim=0))
        if self.representation_type == 'tdex':
            all_repr = torch.concat((tactile_repr, vision_repr), dim=-1)
            return all_repr
        if self.representation_type == 'tactile':
            return tactile_repr 
        if self.representation_type == 'image':
            return vision_repr
        
    def _sample_action(self, all_repr):
        logits, mu, sigma = self.gmm_layer(all_repr) # logits: 1 G, mu: 1 G A, sigma: 1 G A
        # print('logits, mu, sigma shape: ({},{},{})'.format(
        #     logits.shape, mu.shape, sigma.shape
        # ))
        idx = Categorical(logits=logits).sample()  # 1
        # print(f'idx.shape: {idx.shape}')
        sampled_data = Normal(mu, sigma).sample()  # 1 G A
        # print('sampled_data before indexing.shape: {}'.format(
        #     sampled_data.shape
        # ))
        sampled_data = batch_indexing(sampled_data, idx)  # 1 A
        # print('sampled_data after indexing: {}'.format(
        #     sampled_data.shape
        # ))

        return sampled_data

    def get_action(self, tactile_values, recv_robot_state, visualize=False):
        # Get the current visual image
        image = self._get_curr_image().to(self.device)
        all_repr = self._get_all_repr(tactile_values, image)
        print('all_repr.shape: {}'.format(all_repr.shape))
        pred_action = self._sample_action(all_repr).squeeze().detach().cpu().numpy()

        action = dict(
            allegro = pred_action[:16],
            kinova = pred_action[16:]
        )
        
        if visualize:
            self._visualize_state(tactile_values, image)

        self.state_id += 1

        return action

    def _visualize_state(self, tactile_values, image):
        curr_image = self.inv_image_transform(image).detach().cpu().numpy().transpose(1,2,0)
        curr_image_cv2 = cv2.cvtColor(curr_image*255, cv2.COLOR_RGB2BGR)

        tactile_image = self.tactile_img.get_tactile_image_for_visualization(tactile_values)
        dump_whole_state(tactile_values, tactile_image, None, None, title='curr_state', vision_state=curr_image_cv2)
        curr_state = cv2.imread('curr_state.png')
        image_path = os.path.join(self.deployment_dump_dir, f'state_{str(self.state_id).zfill(2)}.png')
        cv2.imwrite(image_path, curr_state)

    def save_deployment(self):
        pass



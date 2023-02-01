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
from torchvision.datasets.folder import default_loader as loader
from torchvision import models

from holobot.constants import *
from holobot.utils.network import ZMQCameraSubscriber
from holobot.robot.allegro.allegro_kdl import AllegroKDL

from tactile_learning.deployment.load_models import load_model
from tactile_learning.deployment.nn_buffer import NearestNeighborBuffer
from tactile_learning.models.knn import KNearestNeighbors, ScaledKNearestNeighbors
from tactile_learning.utils.augmentations import crop_transform
from tactile_learning.utils.constants import *
from tactile_learning.utils.data import load_data
from tactile_learning.utils.tactile_image import get_tactile_image
from tactile_learning.utils.visualization import *
from torchvision.transforms.functional import crop

class DeployBC:
    def __init__(
        self,
        data_path,
        deployment_run_name,
        out_dir, # We will be experimenting with the trained encoders with bc
        robots = ['allegro', 'kinova'],
        view_num = 1,
        representation_type = 'all' # 'tactile, image, all
    ):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"

        torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
        torch.cuda.set_device(0)

        self.robots = robots
        self.vision_view_num = view_num
        self.device = torch.device('cuda:0')
        self._load_bc_models(self.device, out_dir)
        self.representation_type = representation_type

        self.state_id = 0
        self.inv_image_transform = self._get_inverse_image_norm()
        self.deployment_dump_dir = os.path.join('/home/irmak/Workspace/Holo-Bot/deployment_data', deployment_run_name)
        os.makedirs(self.deployment_dump_dir, exist_ok=True)

    def _get_inverse_image_norm(self):
        np_means = np.asarray(VISION_IMAGE_MEANS)
        np_stds = np.asarray(VISION_IMAGE_STDS)

        inv_normalization_transform = T.Compose([
            T.Normalize(mean = [0,0,0], std = 1 / np_stds ), 
            T.Normalize(mean = -np_means, std = [1,1,1])
        ])

        return inv_normalization_transform

    def _load_bc_models(self, device, out_dir):
        cfg = OmegaConf.load(os.path.join(out_dir, '.hydra/config.yaml'))
        image_encoder_path = os.path.join(out_dir, 'models/bc_image_encoder_best.pt')
        self.image_encoder = load_model(cfg, device, image_encoder_path, bc_model_type='image')
        tactile_encoder_path = os.path.join(out_dir, 'models/bc_tactile_encoder_best.pt')
        self.tactile_encoder = load_model(cfg, device, tactile_encoder_path, bc_model_type='tactile')
        last_layer_path = os.path.join(out_dir, 'models/bc_last_layer_best.pt')
        self.last_layer = load_model(cfg, device, last_layer_path, bc_model_type='last_layer')

        # Set up the transforms for tactile and image encoders
        self.image_transform = T.Compose([
            T.Resize((480,640)),
            T.Lambda(self._crop_transform),
            T.ToTensor(),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS),
        ])

        self.dataset_tactile_transform = T.Compose([
            T.Resize(cfg.tactile_image_size),
            T.Lambda(self._clamp_transform), # These are for normalization
            T.Lambda(self._scale_transform)
        ])

        self.tactile_transform = T.Compose([
            T.Resize((224, 224)),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _get_tactile_representation_with_stacked_tactile_encoder(self, tactile_values):
        def _get_stacked_tactile_image(tactile_values):
            tactile_image = torch.FloatTensor(tactile_values)
            tactile_image = tactile_image.view(15,4,4,3) # Just making sure that everything stays the same
            tactile_image = torch.permute(tactile_image, (0,3,1,2))
            tactile_image = tactile_image.reshape(-1,4,4)
            return self.dataset_tactile_transform(tactile_image)

        tactile_image = _get_stacked_tactile_image(tactile_values)
        tactile_image = self.tactile_transform(tactile_image)
        return self.tactile_encoder(tactile_image.unsqueeze(0))

    def _get_alexnet_tactile_representation(self, tactile_values): # This is dependent on
        def _get_whole_hand_tactile_image(tactile_values): 
            # tactile_values: (15,16,3) - turn it into 16,16,3 by concatenating 0z
            tactile_image = torch.FloatTensor(tactile_values)
            tactile_image = F.pad(tactile_image, (0,0,0,0,1,0), 'constant', 0)
            # reshape it to 4x4
            tactile_image = tactile_image.view(16,4,4,3)

            # concat for it have its proper shape
            tactile_image = torch.concat([
                torch.concat([tactile_image[i*4+j] for j in range(4)], dim=0)
                for i in range(4)
            ], dim=1)

            tactile_image = torch.permute(tactile_image, (2,0,1))
            
            return self.dataset_tactile_transform(tactile_image)
        
        tactile_image = _get_whole_hand_tactile_image(tactile_values)
        tactile_image = self.tactile_transform(tactile_image)
        return self.tactile_encoder(tactile_image.unsqueeze(0))

    # def _get_al_repr(self, tactile_)

    def _get_curr_image(self, host='172.24.71.240', port=10005):
        image_subscriber = ZMQCameraSubscriber(
            host = host,
            port = port + self.vision_view_num,
            topic_type = 'RGB'
        )
        image, _ = image_subscriber.recv_rgb_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = im.fromarray(image)
        img = self.image_transform(image)
        return torch.FloatTensor(img)

    def _get_stacked_tactile_image(self, tactile_values):
        tactile_image = torch.FloatTensor(tactile_values)
        tactile_image = tactile_image.view(15,4,4,3) # Just making sure that everything stays the same
        tactile_image = torch.permute(tactile_image, (0,3,1,2))
        tactile_image = tactile_image.reshape(-1,4,4) # Make 45 the channel number 
        return self.tactile_transform(tactile_image)

    def get_action(self, tactile_values, recv_robot_state, visualize=False):
        # Get the current visual image
        image = self._get_curr_image()

        tactile_repr = self._get_alexnet_tactile_representation(tactile_values)
        image_repr = self.image_encoder(image.unsqueeze(dim=0)) # Add a dimension to the first axis so that it could be considered as a batch     
        all_repr = torch.concat((tactile_repr, image_repr), dim=-1)
        if self.representation_type == 'all':
            pred_action = self.last_layer(all_repr)
        elif self.representation_type == 'tactile':
            pred_action = self.last_layer(tactile_repr)
        elif self.representation_type == 'image':
            pred_action = self.last_layer(image_repr)
        pred_action = pred_action.squeeze().detach().cpu().numpy()
        
        action = dict(
            allegro = pred_action[:16],
            kinova = pred_action[16:]
        )
        print('sent action: {}'.format(action))

        if visualize:
            self._visualize_state(tactile_values, image)

        self.state_id += 1

        return action

    def _visualize_state(self, tactile_values, image):
        curr_image = self.inv_image_transform(image).numpy().transpose(1,2,0)
        curr_image_cv2 = cv2.cvtColor(curr_image*255, cv2.COLOR_RGB2BGR)

        tactile_image = self._get_tactile_image_for_visualization(tactile_values)
        dump_whole_state(tactile_values, tactile_image, None, None, title='curr_state', vision_state=curr_image_cv2)
        curr_state = cv2.imread('curr_state.png')
        image_path = os.path.join(self.deployment_dump_dir, f'state_{str(self.state_id).zfill(2)}.png')
        # print('image_path: {}'.format(image_path))
        cv2.imwrite(image_path, curr_state)

    def _get_tactile_image_for_visualization(self, tactile_values):
        def _get_whole_hand_tactile_image(tactile_values): 
            # tactile_values: (15,16,3) - turn it into 16,16,3 by concatenating 0z
            tactile_image = torch.FloatTensor(tactile_values)
            tactile_image = F.pad(tactile_image, (0,0,0,0,1,0), 'constant', 0)
            # reshape it to 4x4
            tactile_image = tactile_image.view(16,4,4,3)

            # concat for it have its proper shape
            tactile_image = torch.concat([
                torch.concat([tactile_image[i*4+j] for j in range(4)], dim=0)
                for i in range(4)
            ], dim=1)

            tactile_image = torch.permute(tactile_image, (2,0,1))
            pre_tactile_transform = T.Compose([
                T.Resize((16,16)),
                T.Lambda(self._clamp_transform),
                T.Lambda(self._scale_transform)
            ])
            return pre_tactile_transform(tactile_image)

        tactile_image = _get_whole_hand_tactile_image(tactile_values)
        tactile_image = T.Resize(224)(tactile_image) # Don't need another normalization
        tactile_image = (tactile_image - tactile_image.min()) / (tactile_image.max() - tactile_image.min())
        return tactile_image 

    def _scale_transform(self, image): # Transform function to map the image between 0 and 1
        image = (image - TACTILE_PLAY_DATA_CLAMP_MIN) / (TACTILE_PLAY_DATA_CLAMP_MAX - TACTILE_PLAY_DATA_CLAMP_MIN)
        return image

    def _clamp_transform(self, image):
        image = torch.clamp(image, min=TACTILE_PLAY_DATA_CLAMP_MIN, max=TACTILE_PLAY_DATA_CLAMP_MAX)
        return image

    def _crop_transform(self, image):
        if self.vision_view_num == 0:
            return crop(image, 0,0,480,480)
        elif self.vision_view_num == 1:
            return crop(image, 0,120,480,480)

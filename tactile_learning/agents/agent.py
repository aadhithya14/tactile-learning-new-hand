# Base class for the agent module
# It will set the expert demos and the encoders
import numpy as np 
import torch

from abc import ABC, abstractmethod

from tactile_learning.models import * 
from tactile_learning.utils import * 
from tactile_learning.tactile_data import *

class Agent(ABC):
    def __init__(
        self,
        data_path,
        expert_demo_nums,
        image_out_dir, image_model_type,
        tactile_out_dir, tactile_model_type,
        view_num, 
        device,
        lr, update_every_steps, stddev_schedule, stddev_clip, features_repeat, # Learning based parts
        **kwargs
    ):
        # Demo based parameters
        self.data_path = data_path
        self.roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
        self.data = load_data(self.roots, demos_to_use=expert_demo_nums)
        self.expert_demo_nums = expert_demo_nums 
        self.device = device
        self.view_num = view_num

        # Learning based parameters
        self.lr = lr
        self.update_every_steps = update_every_steps 
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.features_repeat = features_repeat

        # Get the expert demos and set the encoders
        self._set_image_transform()
        self._set_encoders(
            image_out_dir = image_out_dir, 
            image_model_type = image_model_type,
            tactile_out_dir = tactile_out_dir, 
            tactile_model_type = tactile_model_type
        )
        self._set_expert_demos()

    def _set_encoders(self, image_out_dir, image_model_type, tactile_out_dir, tactile_model_type): 
        _, self.image_encoder, self.image_transform  = init_encoder_info(self.device, image_out_dir, 'image', view_num=self.view_num, model_type=image_model_type)

        tactile_cfg, self.tactile_encoder, _ = init_encoder_info(self.device, tactile_out_dir, 'tactile', view_num=self.view_num, model_type=tactile_model_type)
        tactile_img = TactileImage(
            tactile_image_size = tactile_cfg.tactile_image_size, 
            shuffle_type = None
        )
        tactile_repr_dim = tactile_cfg.encoder.tactile_encoder.out_dim if tactile_model_type == 'bc' else tactile_cfg.encoder.out_dim
        
        self.tactile_repr = TactileRepresentation( # This will be used when calculating the reward - not getting the observations
            encoder_out_dim = tactile_repr_dim,
            tactile_encoder = self.tactile_encoder,
            tactile_image = tactile_img,
            representation_type = 'tdex',
            device = self.device
        )
        
        # Freeze the encoders
        self.image_encoder.eval()
        self.tactile_encoder.eval()
        for param in self.image_encoder.parameters():
            param.requires_grad = False 
        for param in self.tactile_encoder.parameters():
            param.requires_grad = False


    def _set_image_transform(self):
        self.image_act_transform = T.Compose([
            RandomShiftsAug(pad=4),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)
        ])

        self.image_normalize = T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)
        self.inv_image_transform = get_inverse_image_norm() # This is only to be able to 

    def _set_expert_demos(self): # Will stack the end frames back to back
        # We'll stack the tactile repr and the image observations
        self.expert_demos = []
        image_obs = [] 
        tactile_reprs = []
        actions = []
        old_demo_id = -1
        for step_id in range(len(self.data['image']['indices'])): 
            # Set observations
            demo_id, tactile_id = self.data['tactile']['indices'][step_id]
            if (demo_id != old_demo_id and step_id > 0) or (step_id == len(self.data['image']['indices'])-1):
                
                self.expert_demos.append(dict(
                    image_obs = torch.stack(image_obs, 0),
                    tactile_repr = torch.stack(tactile_reprs, 0),
                    actions = np.stack(actions, 0)
                ))
                image_obs = [] 
                tactile_reprs = []
                actions = []

            tactile_value = self.data['tactile']['values'][demo_id][tactile_id]
            tactile_repr = self.tactile_repr.get(tactile_value, detach=False)

            _, image_id = self.data['image']['indices'][step_id]
            image = load_dataset_image(
                data_path = self.data_path, 
                demo_id = demo_id, 
                image_id = image_id,
                view_num = self.view_num,
                transform = self.image_transform
            )

            # Set actions
            _, allegro_action_id = self.data['allegro_actions']['indices'][step_id]
            allegro_action = self.data['allegro_actions']['values'][demo_id][allegro_action_id]
            _, kinova_id = self.data['kinova']['indices'][step_id]
            kinova_action = self.data['kinova']['values'][demo_id][kinova_id]
            demo_action = np.concatenate([allegro_action, kinova_action], axis=-1)

            image_obs.append(image)
            tactile_reprs.append(tactile_repr)
            actions.append(demo_action)

            old_demo_id = demo_id

    def _get_policy_reprs_from_obs(self, image_obs, tactile_repr, features, representation_types):
         # Get the representations
        reprs = []
        if 'image' in representation_types:
            # Current representations
            image_obs = self.image_act_transform(image_obs.float()).to(self.device)
            image_reprs = self.image_encoder(image_obs)
            reprs.append(image_reprs)

        if 'tactile' in representation_types:
            tactile_reprs = tactile_repr.to(self.device) # This will give all the representations of one batch
            reprs.append(tactile_reprs)

        if 'features' in representation_types:
            repeated_features = features.repeat(1, self.features_repeat)
            reprs.append(repeated_features.to(self.device))

        return torch.concat(reprs, axis=-1) # Concatenate the representations to get the final representations

    @abstractmethod
    def update(self, **kwargs):
        pass 

    @abstractmethod
    def act(self, **kwargs): 
        pass

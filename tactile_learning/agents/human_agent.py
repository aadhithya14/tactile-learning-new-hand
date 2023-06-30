import glob
import numpy as np
import torch

from tqdm import tqdm
from tactile_learning.utils import load_human_data, load_dataset_image

from .agent import Agent 

# Implementation for a human based data agent 
# expert_demos will be overwritten 

class HumanAgent(Agent):
    def __init__(
        self,
        data_path,
        expert_demo_nums,
        image_out_dir, image_model_type,
        tactile_out_dir, tactile_model_type,
        view_num, 
        device,
        lr, update_every_steps, stddev_schedule, stddev_clip, features_repeat, 
        experiment_name, # Learning based parts
        **kwargs
    ):
        
        super().__init__(
            data_path,
            expert_demo_nums,
            image_out_dir, image_model_type,
            tactile_out_dir, tactile_model_type,
            view_num, 
            device,
            lr, update_every_steps, stddev_schedule, stddev_clip, features_repeat, 
            experiment_name, # Learning based parts
            **kwargs
        )

    def _set_expert_demos(self): # Will stack the end frames back to back
        # We'll stack the tactile repr and the image observations
        self.expert_demos = []
        image_obs = [] 
        keypoints = []
        old_demo_id = -1
        pbar = tqdm(total=len(self.data['image']['indices']))
        for step_id in range(len(self.data['image']['indices'])): 
            # Set observations
            demo_id, keypoint_id = self.data['keypoint']['indices'][step_id]
            if (demo_id != old_demo_id and step_id > 0) or (step_id == len(self.data['image']['indices'])-1):
                
                self.expert_demos.append(dict(
                    image_obs = torch.stack(image_obs, 0),
                    keypoints = np.stack(keypoints, 0)
                    # actions = np.stack(actions, 0)
                ))
                image_obs = [] 
                keypoints = []

            _, image_id = self.data['image']['indices'][step_id]
            image = load_dataset_image(
                data_path = self.data_path, 
                demo_id = demo_id, 
                image_id = image_id,
                view_num = self.view_num,
                transform = self.image_transform
            )

            # Set the keypoints
            _, keypoint_id = self.data['keypoint']['indices'][step_id]
            keypoint = self.data['keypoint']['values'][demo_id][keypoint_id]

            image_obs.append(image)
            keypoints.append(keypoint)

            old_demo_id = demo_id

            pbar.update(1)
            pbar.set_description('Setting the expert demos ')

        pbar.close()

    def _set_data(self, data_path, expert_demo_nums):
        self.data_path = data_path 
        self.expert_demo_nums = expert_demo_nums
        self.roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
        self.data = load_human_data(self.roots, demos_to_use=expert_demo_nums)

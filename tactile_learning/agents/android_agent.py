import glob
import numpy as np
import torch

from tqdm import tqdm
from tactile_learning.utils import load_human_data, load_dataset_image

from .agent import Agent 
from .mix_agent import MixAgent
from tactile_learning.utils import * 

# Implementation for a human based data agent 
# expert_demos will be overwritten 

class AndroidAgent(MixAgent):
    def __init__(
        self,
        data_path,
        robot_data_path, 
        expert_demo_nums, robot_demo_nums,
        image_out_dir, image_model_type,
        tactile_out_dir, tactile_model_type,
        view_num, 
        device,
        lr, update_every_steps, stddev_schedule, stddev_clip, features_repeat, 
        experiment_name, # Learning based parts
        **kwargs
    ):
        
        super().__init__(
            data_path, robot_data_path, 
            expert_demo_nums, robot_demo_nums,
            image_out_dir, image_model_type,
            tactile_out_dir, tactile_model_type,
            view_num, 
            device,
            lr, update_every_steps, stddev_schedule, stddev_clip, features_repeat, 
            experiment_name, # Learning based parts
            **kwargs
        )
        self.robot_data_path = robot_data_path
        self.robot_demo_nums = robot_demo_nums


    def _set_expert_demos(self): # Will stack the end frames back to back
        # We'll stack the tactile repr and the image observations
        self.expert_demos = []
        self.robot_demos = []
        tactile_reprs = [] # for robot
        image_obs = []  # for human
        robot_image_obs = []
        keypoints = []
        actions = [] # for robot
        old_demo_id = -1
        pbar = tqdm(total=len(self.data['image']['indices']))
        for step_id in range(len(self.data['image']['indices'])): 
            # Set observations
            demo_id, keypoint_id = self.data['keypoint']['indices'][step_id]
            robot_demo_id, tactile_id = self.robot_data['tactile']['indices'][step_id]
            if (demo_id != old_demo_id and step_id > 0) or (step_id == len(self.data['image']['indices'])-1):
                
                self.expert_demos.append(dict(
                    image_obs = torch.stack(image_obs, 0),
                    keypoints = np.stack(keypoints, 0)
                    # actions = np.stack(actions, 0)
                ))
                image_obs = [] 
                keypoints = []

                self.robot_demos.append(dict(
                    image_obs = torch.stack(robot_image_obs, 0),
                    tactile_repr = torch.stack(tactile_reprs, 0),
                    actions = np.stack(actions, 0)
                ))
                robot_image_obs = [] 
                tactile_reprs = []
                actions = []

            # set human demo
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

            # pbar.update(1)
            # pbar.set_description('Setting the expert demos ')

            # set robot demo
            tactile_value = self.robot_data['tactile']['values'][robot_demo_id][tactile_id]
            tactile_repr = self.tactile_repr.get(tactile_value, detach=False)

            _, robot_image_id = self.robot_data['image']['indices'][step_id]
            robot_image = load_dataset_image(
                data_path = self.robot_data_path, 
                demo_id = robot_demo_id, 
                image_id = robot_image_id,
                view_num = self.view_num,
                transform = self.image_transform
            )

            # Set actions
            _, allegro_action_id = self.robot_data['allegro_actions']['indices'][step_id]
            allegro_action = self.robot_data['allegro_actions']['values'][robot_demo_id][allegro_action_id]
            _, kinova_id = self.robot_data['kinova']['indices'][step_id]
            kinova_action = self.robot_data['kinova']['values'][robot_demo_id][kinova_id]
            demo_action = np.concatenate([allegro_action, kinova_action], axis=-1)

            robot_image_obs.append(robot_image)
            tactile_reprs.append(tactile_repr)
            actions.append(demo_action)

            # old_demo_id = demo_id

            pbar.update(1)
            pbar.set_description('Setting the expert demos ')

        pbar.close()


    def _set_data(self, data_path, expert_demo_nums, robot_data_path, robot_demo_nums):
        self.data_path = data_path 
        self.expert_demo_nums = expert_demo_nums
        self.roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
        self.data = load_human_data(self.roots, demos_to_use=expert_demo_nums)

        #add self.robot_data here
        self.robot_roots = sorted(glob.glob(f'{robot_data_path}/demonstration_*'))
        self.robot_data = load_data(self.robot_roots, demos_to_use=robot_demo_nums)
        
        
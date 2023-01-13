# Script to use some of the deployment wrappers and apply the actions
import hydra
import matplotlib.pyplot as plt
import numpy as np 
import os
import pickle
import torch 
import sys


# from holobot.components.deploy.commander import DeployAPI
from holobot_api.api import DeployAPI
from holobot.utils.timer import FrequencyTimer
from holobot.constants import ALLEGRO_BOX_HANDLE_LIFTING_THUMB_VALUES
from omegaconf import DictConfig, OmegaConf

class Deployer:
    def __init__(self, cfg, deployed_module, robots):
        self.module = deployed_module
        self.robots = robots
        required_data = {
            'rgb_idxs': [0],
            'depth_idxs': [0]
        }
        self.deploy_api = DeployAPI(
            host_address = '172.24.71.240',
            required_data = required_data
        )
        self.cfg = cfg
        self.data_path = cfg.data_path
        self.device = torch.device('cuda:0')
        self.frequency_timer = FrequencyTimer(cfg.frequency)

    def _load_stats(self):
        # Load the mean and std of the allegro hand
        with open(os.path.join(self.data_path, 'allegro_stats.pkl'), 'rb') as f:
            allegro_stats = pickle.load(f)
            print('allegro_stats.shape: {}'.format(allegro_stats.shape))
        self._allegro_mean, self._allegro_std = allegro_stats[0], allegro_stats[1]

        # Load the mean and std of the tactile sensor
        with open(os.path.join(self.data_path, 'tactile_stats.pkl'), 'rb') as f: 
            tactile_stats = pickle.load(f) 
            print('tactile_stats.shape: {}'.format(tactile_stats.shape))
        self._tactile_mean, self._tactile_std = tactile_stats[0], tactile_stats[1]

    def _normalize_allegro_state(self, allegro_state):
        return torch.FloatTensor((allegro_state - self._allegro_mean) / self._allegro_std).to(self.device)

    def _normalize_tactile_state(self, tactile_state):
        return torch.FloatTensor((tactile_state - self._tactile_mean) / self._tactile_std).to(self.device)

    def solve(self):
        sys.stdin = open(0) # To get inputs while spawning multiple processes

        while True:
            
            try:

                if self.cfg['loop']:
                    self.frequency_timer.start_loop()

                print('\n***************************************************************')
                print('\nGetting state information...') 

                # Get the robot state and the tactile info
                robot_state = self.deploy_api.get_robot_state() 
                sensor_state = self.deploy_api.get_sensor_state()

                allegro_joint_pos = robot_state['allegro']['position']
                tactile_info = sensor_state['xela']['sensor_values']
                send_robot_state = dict(
                    allegro = allegro_joint_pos
                )
                if 'kinova' in self.cfg.robots:
                    kinova_state = robot_state['kinova']
                    send_robot_state['kinova'] = kinova_state

                pred_action = self.module.get_action(
                    tactile_info,
                    send_robot_state,
                    visualize=self.cfg['visualize']
                )
                print('\nPredicted action: {}'.format(pred_action))

                if not self.cfg['loop']:
                    register = input('\nPress a key to perform the action...')

                action_dict = dict() 
                action_dict['allegro'] = pred_action['allegro'] # Should be a numpy array
                action_dict['kinova'] = pred_action['kinova']
                self.deploy_api.send_robot_action(action_dict)

                if self.cfg['loop']: 
                    self.frequency_timer.end_loop()

            except KeyboardInterrupt:
                self.module.save_deployment() # This is supposed to save all the representaitons and run things 

@hydra.main(version_base=None, config_path='tactile_learning/configs', config_name='deploy')
def main(cfg : DictConfig) -> None:
    if cfg.object == 'box_handle_lifting':
        set_thumb_values = ALLEGRO_BOX_HANDLE_LIFTING_THUMB_VALUES
    else:
        set_thumb_values = None
    deploy_module = hydra.utils.instantiate(
        cfg.deploy_module,
        set_thumb_values = set_thumb_values,
        data_path = cfg.data_path
    )
    print('deploy_module: {}'.format(deploy_module))
    deployer = Deployer(cfg, deploy_module, cfg.deploy_module.robots)
    deployer.solve()

if __name__ == '__main__':
    main()

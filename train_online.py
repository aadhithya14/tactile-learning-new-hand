# This script is used to train the policy online
import os
import hydra

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm 

# Custom imports 
# from tactile_learning.datasets import get_dataloaders

# from tactile_learning.learners import init_learner
from tactile_learning.datasets import *
from tactile_learning.environments import MockEnv
from tactile_learning.models import *
from tactile_learning.utils import *


class Workspace:
    def __init__(self, cfg):
        # Set the variables
        self.cfg = cfg
        set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.data_path = cfg.data_path

        # Initialize hydra
        self.hydra_dir = HydraConfig.get().run.dir

        # Run the setup - this should start the replay buffer and the environment
        self._env_setup()
        self._encoder_setup(cfg) # Get the image and tactile encoder/representation module

        # Get the mock environment observations and the tactile representations - this will
        # be changed as we use the actual environment
        self.data_path = cfg.data_path
        self.roots = sorted(glob.glob(f'{cfg.data_path}/demonstration_*'))
        self.mock_data = load_data(self.roots, demos_to_use=cfg.mock_demo_nums)
        self._set_mock_demos() # Get the mock demo observation and representations
        self.mock_env = MockEnv(self.mock_episodes)

        # self.agent = hydra.utils.instantiate(cfg.agent)
        self._initialize_agent()

        # TODO: Timer? - should we set a timer - I think we need this for real world demos
        self._global_step = 0 
        self._global_episode = 0

        # Set the logger right before the training
        self._set_logger(cfg)

    def _initialize_agent(self):
        action_spec = self.mock_env.action_spec() # TODO: Change this to the training env
        self.cfg.agent.action_shape = action_spec.shape
        self.agent = hydra.utils.instantiate(self.cfg.agent)

    def _set_logger(self, cfg):
        wandb_exp_name = '-'.join(self.hydra_dir.split('/')[-2:])
        self.logger = Logger(cfg, wandb_exp_name, out_dir=self.hydra_dir)

    def _encoder_setup(self, cfg):
        image_cfg, self.image_encoder, self.image_transform = init_encoder_info(self.device, cfg.image_out_dir, 'image')
        self.inv_image_transform = get_inverse_image_norm() 

        tactile_cfg, self.tactile_encoder, _ = init_encoder_info(self.device, cfg.tactile_out_dir, 'tactile')
        tactile_img = TactileImage(
            tactile_image_size = tactile_cfg.tactile_image_size, 
            shuffle_type = None
        )
        self.tactile_repr = TactileRepresentation( # This will be used when calculating the reward - not getting the observations
            encoder_out_dim = tactile_cfg.encoder.out_dim,
            tactile_encoder = self.tactile_encoder,
            tactile_image = tactile_img,
            representation_type = 'tdex'
        )
        self.view_num = 1

        # Freeze the encoders
        self.image_encoder.eval()
        self.tactile_encoder.eval()
        for param in self.image_encoder.parameters():
            param.requires_grad = False 
        for param in self.tactile_encoder.parameters():
            param.requires_grad = False

    def _env_setup(self):
        pass # TODO - here set the data_specs and everything

    def _set_mock_demos(self):
        # We'll stack the tactile repr and the image observations
        end_of_demos = np.zeros(len(self.mock_data['image']['indices']))
        for step_id in range(len(self.mock_data['image']['indices'])): 
            demo_id, tactile_id = self.mock_data['tactile']['indices'][step_id]

            # Check if the demo id stays the same or not
            if step_id > 1:
                if demo_id != prev_demo_id:
                    end_of_demos[step_id-1] = 1 # 1 for steps where it's the end of an episode

            tactile_value = self.mock_data['tactile']['values'][demo_id][tactile_id]
            tactile_repr = self.tactile_repr.get(tactile_value, detach=False)

            _, image_id = self.mock_data['image']['indices'][step_id]
            image = load_dataset_image(
                data_path = self.data_path, 
                demo_id = demo_id, 
                image_id = image_id,
                view_num = self.view_num,
                transform = self.image_transform
            )

            if step_id == 0:
                tactile_reprs = tactile_repr.unsqueeze(0)
                image_obs = image.unsqueeze(0)
            else:
                image_obs = torch.concat([image_obs, image.unsqueeze(0)], dim=0)
                tactile_reprs = torch.concat([tactile_reprs, tactile_repr.unsqueeze(0)], dim=0)

            prev_demo_id = demo_id

        end_of_demos[-1] = 1
        
        self.mock_episodes = dict(
            image_obs = image_obs, 
            tactile_reprs = tactile_reprs,
            end_of_demos = end_of_demos # end_of_demos[time_step] will be 1 if this is end of an episode  
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode
    
    # Main online training code - this will be giving the rewards only for now
    def train_online(self):
        # Set the predicates for training
        train_until_step = Until(self.cfg.num_train_frames)
        seed_until_step = Until(self.cfg.num_seed_frames)
        eval_every_step = Every(self.cfg.eval_every_frames) # Evaluate in every these steps

        episode_step, episode_reward = 0, 0

        # TODO: Get the actions, observations and time_steps as input from the train environment in the future - lines 189 - 202
        # Reset step implementations 
        time_steps = list() 
        observations = dict(
            image_obs = list(),
            tactile_repr = list()
        )
        time_step = self.mock_env.reset() # TODO: turn this into actual environment
        
        # print(f'RESETTED TIME_STEP: {time_step}')
        time_steps.append(time_step)
        for obs_type in time_step.observation.keys():
            observations[obs_type].append(time_step.observation[obs_type])

        if self.agent.auto_rew_scale:
            self.agent.sinkhorn_rew_scale = 1. # This will be set after the first episode

        # metrics = None - TODO: Log these afterwards
        while train_until_step(self.global_step): # We're going to behave as if we act and the observations and the representations are coming from the mock_demo but all the rest should be the same
            
            # At the end of an episode actions
            if time_step.last(): # TODO: This could require more checks in the real world
                
                self._global_episode += 1 # Episode has been finished
                
                # Make each element in the observations to be a new array
                for obs_type in observations.keys():
                    observations[obs_type] = torch.stack(observations[obs_type], 0)

                new_rewards = self.agent.ot_rewarder(
                    episode_obs = observations
                )
                new_rewards_sum = np.sum(new_rewards)
                print(f'REWARD = {new_rewards_sum}')

                # Scale the rewards to -10 for the first demo
                if self.agent.auto_rew_scale:
                    if self._global_episode == 1:
                        self.agent.sinkhorn_rew_scale = self.agent.sinkhorn_rew_scale * self.agent.auto_rew_scale_factor / float(
                            np.abs(new_rewards_sum))
                        new_rewards = self.agent.ot_rewarder(
                            episode_obs = observations
                        )
                        new_rewards_sum = np.sum(new_rewards)

                # Update the reward in the timesteps accordingly
                for i, elt in enumerate(time_steps):
                    # elt = elt._replace( - NOTE: I am not sure why we have this?
                    #     observation=time_steps[i].observation[self.cfg.obs_type])
                    elt = elt._replace(reward=new_rewards[i])

                    # TODO: Replay Buffer should be used here - replay_storage.add should happen right here

                # Log
                self.logger.log({
                    'episode': self.global_episode,
                    'episode_reward': episode_reward 
                }) 
                self.logger.log({ # NOTE: This basically will be the only important reward lol
                    'episode': self.global_episode,
                    'imitation_reward': new_rewards_sum
                })
                

                # Reset the environment at the end of the episode
                time_steps = list()
                observations = dict(
                    image_obs = list(),
                    tactile_repr = list()
                ) 

                x = input("Press Enter to continue... after reseting env")

                time_step = self.mock_env.reset()
                time_steps.append(time_step)
                for obs_type in time_step.observation.keys():
                    observations[obs_type].append(time_step.observation[obs_type])

                episode_step, episode_reward = 0, 0

            # Get the action - TODO: This will be actual sampled action in the real running
            # with torch.no_grad(), utils.eval_mode(self.agent):
            #     action, base_action = self.agent.act(
            #         time_step.observation[self.cfg.obs_type],
            #         self.global_step,
            #         eval_mode=False
            # )

            # Training - updating the agents - TODO: For now we're not running this
            if not seed_until_step(self.global_step):
                pass
                # Update - TODO: Should log the metrics as well
                # metrics = self.agent.update(self.replay_iter, self.expert_replay_iter, self._ssl_replay_iter, 
                #                             self.global_step, self.cfg.bc_regularize)
                # self.logger.log_metrics(...) -> TODO: Implement this for wandb
             
            # Take the environment steps    
            time_step = self.mock_env.step(action=None) # TODO: Give an actual action
            # Print the time_step action
            # print('Episode Step: {}, Time Step: {}'.format(
            #     episode_step, time_step
            # ))
            episode_reward += time_step.reward

            time_steps.append(time_step)
            for obs_type in time_step.observation.keys():
                    observations[obs_type].append(time_step.observation[obs_type])

            episode_step += 1
            self._global_step += 1 


@hydra.main(version_base=None, config_path='tactile_learning/configs', config_name='train_online')
def main(cfg: DictConfig) -> None:
    workspace = Workspace(cfg)
    workspace.train_online()


if __name__ == '__main__':
    main()
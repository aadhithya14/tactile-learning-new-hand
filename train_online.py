# This script is used to train the policy online
import datetime
import os
import hydra

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dm_env import specs
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm 

from PIL import Image
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
        self.work_dir = Path.cwd()
        self.cfg = cfg
        set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.data_path = cfg.data_path

        # Initialize hydra
        self.hydra_dir = HydraConfig.get().run.dir

        # Run the setup - this should start the replay buffer and the environment
        tactile_repr_dim = self._encoder_setup(cfg) # Get the image and tactile encoder/representation module
        self.data_path = cfg.data_path
        self.mock_env = cfg.mock_env
        self._env_setup(tactile_repr_dim) # Should be set here

        # If load the snapshot rather than agent
        # self.load_snapshot = cfg.load_snapshot 

        # self.agent = hydra.utils.instantiate(cfg.agent)
        self._initialize_agent()

        # TODO: Timer? - should we set a timer - I think we need this for real world demos
        self._global_step = 0 
        self._global_episode = 0

        # Set the logger right before the training
        self._set_logger(cfg)

    def _initialize_agent(self):
        action_spec = self.train_env.action_spec()
        self.cfg.agent.action_shape = action_spec.shape

        self.agent = hydra.utils.instantiate(self.cfg.agent)
        self.agent.initialize_modules(
            base_policy_cfg = self.cfg.base_policy,
            rewarder_cfg = self.cfg.rewarder
        )

    def _set_logger(self, cfg):
        if self.cfg.log:
            wandb_exp_name = '-'.join(self.hydra_dir.split('/')[-2:])
            self.logger = Logger(cfg, wandb_exp_name, out_dir=self.hydra_dir)

    def _encoder_setup(self, cfg):
        print('cfg.image_model_type: {}'.format(cfg.image_model_type))
        image_cfg, self.image_encoder, self.image_transform = init_encoder_info(self.device, cfg.image_out_dir, 'image', view_num=cfg.camera_num, model_type=cfg.image_model_type)
        self.inv_image_transform = get_inverse_image_norm() 

        tactile_cfg, self.tactile_encoder, _ = init_encoder_info(self.device, cfg.tactile_out_dir, 'tactile', view_num=cfg.camera_num, model_type=cfg.tactile_model_type)
        tactile_img = TactileImage(
            tactile_image_size = tactile_cfg.tactile_image_size, 
            shuffle_type = None
        )
        tactile_repr_dim = tactile_cfg.encoder.tactile_encoder.out_dim if cfg.tactile_model_type == 'bc' else tactile_cfg.encoder.out_dim
        self.tactile_repr = TactileRepresentation( # This will be used when calculating the reward - not getting the observations
            encoder_out_dim = tactile_repr_dim,
            tactile_encoder = self.tactile_encoder,
            tactile_image = tactile_img,
            representation_type = 'tdex'
        )
        self.view_num = 1

        self.image_episode_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)
        ])

        # Freeze the encoders
        self.image_encoder.eval()
        self.tactile_encoder.eval()
        for param in self.image_encoder.parameters():
            param.requires_grad = False 
        for param in self.tactile_encoder.parameters():
            param.requires_grad = False

        return tactile_repr_dim # Should return the tactile representation dimension

    def _env_setup(self, tactile_repr_dim):

        if self.mock_env:
            self.roots = sorted(glob.glob(f'{self.data_path}/demonstration_*'))
            self.mock_data = load_data(self.roots, demos_to_use=self.cfg.mock_demo_nums)
            self._set_mock_demos() # Get the mock demo observation and representations
            self.train_env = MockEnv(self.mock_episodes)
        else:
            self.train_env = hydra.utils.call( # If not call the actual interaction environment
                self.cfg.suite.task_make_fn,
                tactile_dim = tactile_repr_dim
            )

        # Create replay buffer
        data_specs = [
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array(self.train_env.action_spec().shape, self.train_env.action_spec().dtype, 'base_action'),
            specs.Array((1,), np.float32, 'reward'), 
            specs.Array((1,), np.float32, 'discount')
        ]

        if self.cfg.buffer_path is None:
            replay_dir = self.work_dir / 'buffer' / self.cfg.experiment
        else:
            replay_dir = self.work_dir / 'buffer' / self.cfg.buffer_path
        
        self.replay_storage = ReplayBufferStorage(
            data_specs = data_specs,
            replay_dir = replay_dir # All the experiments are saved under same name
        )

        self.replay_loader = make_replay_loader(
            replay_dir = replay_dir,
            max_size = self.cfg.replay_buffer_size,
            batch_size = self.cfg.batch_size,
            num_workers = self.cfg.replay_buffer_num_workers,
            nstep = self.cfg.nstep,
            save_snapshot = self.cfg.suite.save_snapshot,
            discount = self.cfg.suite.discount
        )

        self._replay_iter = None
        if self.cfg.bc_regularize: # NOTE: If we use bc regularize you should create an expert replay buffer
            self.expert_replay_iter = None
        
        if self.cfg.evaluate:
            self.eval_video_recorder = TrainVideoRecorder( # It is the same recorder for our case
                save_dir = Path(self.work_dir) / 'online_training_outs/eval_video/videos' / self.cfg.experiment if self.cfg.save_eval_video else None,
                root_dir = None)
        self.train_video_recorder = TrainVideoRecorder(
            save_dir = Path(self.work_dir) / 'online_training_outs/train_video/videos' / self.cfg.experiment if self.cfg.save_train_video else None,
            root_dir = None)

    def _set_mock_demos(self):
        # We'll stack the tactile repr and the image observations
        end_of_demos = np.zeros(len(self.mock_data['image']['indices']))
        demo_nums = []

        # Add the number of demos for informational logging
        demo_id, _ = self.mock_data['tactile']['indices'][0]
        root = self.roots[demo_id]
        demo_num = int(root.split('/')[-1].split('_')[-1])
        demo_nums.append(demo_num)

        for step_id in range(len(self.mock_data['image']['indices'])): 
            demo_id, tactile_id = self.mock_data['tactile']['indices'][step_id]

            # Check if the demo id stays the same or not
            if step_id > 1:
                if demo_id != prev_demo_id:
                    end_of_demos[step_id-1] = 1 # 1 for steps where it's the end of an episode
                    root = self.roots[demo_id]
                    demo_num = int(root.split('/')[-1].split('_')[-1])
                    demo_nums.append(demo_num)

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
            end_of_demos = end_of_demos, # end_of_demos[time_step] will be 1 if this is end of an episode  
            demo_nums = demo_nums
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode
    
    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter
    
    def save_snapshot(self, save_step=False, eval=False):
        snapshot = self.work_dir / 'weights'
        snapshot.mkdir(parents=True, exist_ok=True)
        if eval:
            snapshot = snapshot / ('snapshot_eval.pt' if not save_step else f'snapshot_{self.global_step}_eval.pt')
        else:
            snapshot = snapshot / ('snapshot.pt' if not save_step else f'snapshot_{self.global_step}.pt')
        keys_to_save = ['_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open('wb') as f:
            torch.save(payload, f)
                
    def load_snapshot(self, snapshot):
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            # print('k: {}, v: {}'.format(k, v))
            if k not in self.__dict__:
                agent_payload[k] = v
        # self.agent.load_snapshot(agent_payload)
        self.agent.load_snapshot_eval(agent_payload) # NOTE: Not sure why they stopped running this

    def _add_time_step(self, time_step, time_steps, observations):
        pil_image_obs = Image.fromarray(np.transpose(time_step.observation['pixels'], (1,2,0)), 'RGB')
        transformed_image_obs = self.image_episode_transform(pil_image_obs)

        time_steps.append(time_step)
        observations['image_obs'].append(transformed_image_obs)
        observations['tactile_repr'].append(torch.FloatTensor(time_step.observation['tactile']))
        observations['features'].append(torch.FloatTensor(time_step.observation['features']))
 
        return time_steps, observations

    def eval(self, evaluation_step):
        step, episode = 0, 0
        eval_until_episode = Until(self.cfg.num_eval_episodes)
        while eval_until_episode(episode):
            episode_step = 0
            is_episode_done = False
            print(f"Eval Episode {episode}")
            time_steps = list() 
            observations = dict(
                image_obs = list(),
                tactile_repr = list(),
                features = list()
            )
            time_step = self.train_env.reset()
            time_steps, observations = self._add_time_step(time_step, time_steps, observations)
            self.eval_video_recorder.init(time_step.observation['pixels'])

            while not (time_step.last() or is_episode_done):
                with torch.no_grad(), utils.eval_mode(self.agent):

                    action, base_action, is_episode_done, metrics = self.agent.act(
                        obs = dict(
                            image_obs = torch.FloatTensor(time_step.observation['pixels']),
                            tactile_repr = torch.FloatTensor(time_step.observation['tactile']),
                            features = torch.FloatTensor(time_step.observation['features'])
                        ),
                        global_step = self.global_step, 
                        episode_step = episode_step,
                        eval_mode = True # When set to true this will return the mean of the offsets learned from the model
                    )
                time_step = self.train_env.step(action, base_action)
                time_steps, observations = self._add_time_step(time_step, time_steps, observations)
                self.eval_video_recorder.record(time_step.observation['pixels'])
                step += 1
                episode_step += 1
                
                if self.cfg.log:
                    self.logger.log_metrics(metrics, self.global_frame, 'global_frame')

            episode += 1
            x = input("Press Enter to continue... after reseting env")

            self.eval_video_recorder.save(f'{self.cfg.object}_eval_{evaluation_step}_{episode}.mp4')
        
        # Reset env
        self.train_env.reset()

    # Main online training code - this will be giving the rewards only for now
    def train_online(self):
        # Set the predicates for training
        train_until_step = Until(self.cfg.num_train_frames)
        seed_until_step = Until(self.cfg.num_seed_frames)
        eval_every_step = Every(self.cfg.eval_every_frames) # Evaluate in every these steps

        episode_step, episode_reward = 0, 0

        # Reset step implementations 
        time_steps = list() 
        observations = dict(
            image_obs = list(),
            tactile_repr = list(),
            features = list()
        )
        time_step = self.train_env.reset()
        
        self.episode_id = 0
        time_steps, observations = self._add_time_step(time_step, time_steps, observations)

        if self.agent.auto_rew_scale:
            self.agent.sinkhorn_rew_scale = 1. # This will be set after the first episode

        self.train_video_recorder.init(time_step.observation['pixels'])
        metrics = dict() 
        is_episode_done = False
        while train_until_step(self.global_step): # We're going to behave as if we act and the observations and the representations are coming from the mock_demo but all the rest should be the same
            
            # At the end of an episode actions
            if time_step.last() or is_episode_done:
                
                self._global_episode += 1 # Episode has been finished
                
                # Make each element in the observations to be a new array
                for obs_type in observations.keys():
                    observations[obs_type] = torch.stack(observations[obs_type], 0)

                # Get the rewards
                new_rewards = self.agent.ot_rewarder(
                    episode_obs = observations,
                    episode_id = self.global_episode,
                    visualize = self.cfg.save_train_cost_matrices,
                    exponential_weight_init = self.cfg.exponential_weight_init
                )
                new_rewards_sum = np.sum(new_rewards)

                # Scale the rewards to -10 for the first demo
                if self.agent.auto_rew_scale:
                    if self._global_episode == 1:
                        self.agent.sinkhorn_rew_scale = self.agent.sinkhorn_rew_scale * self.agent.auto_rew_scale_factor / float(
                            np.abs(new_rewards_sum))
                        new_rewards = self.agent.ot_rewarder(
                            episode_obs = observations,
                            episode_id = self.global_episode,
                            visualize = False,
                            exponential_weight_init = self.cfg.exponential_weight_init
                        )
                        new_rewards_sum = np.sum(new_rewards)
   
                print(f'REWARD = {new_rewards_sum}')
                ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
                self.train_video_recorder.save(f'{ts}_e{self.global_episode}_f{self.global_frame}_r{round(new_rewards_sum,2)}.mp4')
                
                if self.mock_env:
                    self.episode_id = (self.episode_id+1) % len(self.mock_episodes['demo_nums'])

                # Update the reward in the timesteps accordingly
                obs_length = len(time_steps)
                avg_reward = new_rewards_sum / obs_length
                for i, elt in enumerate(time_steps):
                    if i > (obs_length - self.cfg.reward_matching_steps):
                        new_reward = new_rewards[self.cfg.reward_matching_steps - (obs_length - i)]
                        elt = elt._replace(reward=new_reward) # Update the reward of the object accordingly
                    self.replay_storage.add(elt, last = (i == len(time_steps) - 1))

                # Log
                if self.cfg.log:
                    metrics = {
                        'imitation_reward': new_rewards_sum,
                        'episode_reward': episode_reward
                    }
                    self.logger.log_metrics(metrics, time_step = self.global_episode, time_step_name = 'global_episode')

                # Reset the environment at the end of the episode
                time_steps = list()
                observations = dict(
                    image_obs = list(),
                    tactile_repr = list(),
                    features = list()
                ) 

                x = input("Press Enter to continue... after reseting env")

                time_step = self.train_env.reset()
                time_steps, observations = self._add_time_step(time_step, time_steps, observations)

                # Checkpoint saving and visualization
                self.train_video_recorder.init(time_step.observation['pixels'])
                if self.cfg.suite.save_snapshot:
                    self.save_snapshot()

                episode_step, episode_reward = 0, 0

            # Get the action
            with torch.no_grad(), eval_mode(self.agent):
                if self.mock_env:
                    action, base_action = self.agent.mock_act(
                        time_step.observation,
                        step = self.global_step,
                        max_step = self.train_env.spec.max_episode_steps
                        # self.global_step,
                        # eval_mode=False 
                    )
                else:
                    action, base_action, is_episode_done, metrics = self.agent.act(
                        obs = dict(
                            image_obs = torch.FloatTensor(time_step.observation['pixels']),
                            tactile_repr = torch.FloatTensor(time_step.observation['tactile']),
                            features = torch.FloatTensor(time_step.observation['features'])
                        ),
                        global_step = self.global_step, 
                        episode_step = episode_step,
                        eval_mode = False
                    )
                    if self.cfg.log:
                        self.logger.log_metrics(metrics, self.global_frame, 'global_frame')

            print('STEP: {}'.format(self.global_step))
            print('---------')

            # Training - updating the agents 
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(
                    replay_iter = self.replay_iter,
                    step = self.global_step,
                    bc_regularize = self.cfg.bc_regularize,
                    expert_replay_iter = None, # These could be used in the future
                    ssl_replay_iter = None 
                )
                if self.cfg.log:
                    self.logger.log_metrics(metrics, self.global_frame, 'global_frame')

            if self.cfg.evaluate and eval_every_step(self.cfg.eval_every_frames):
                self.eval(evaluation_step = int(self.global_step/self.cfg.eval_every_frames))
             
            # Take the environment steps    
            time_step = self.train_env.step(action, base_action)
            episode_reward += time_step.reward

            time_steps, observations = self._add_time_step(time_step, time_steps, observations)

            # Record and increase the steps
            self.train_video_recorder.record(time_step.observation['pixels']) # NOTE: Should we do env.render()? 
            episode_step += 1
            self._global_step += 1 

@hydra.main(version_base=None, config_path='tactile_learning/configs', config_name='train_online')
def main(cfg: DictConfig) -> None:
    workspace = Workspace(cfg)

    if cfg.load_snapshot:
        snapshot = Path(cfg.snapshot_weight)
        print(f'Resuming the snapshot: {snapshot}')    
        workspace.load_snapshot(snapshot)

    workspace.train_online()


if __name__ == '__main__':
    main()
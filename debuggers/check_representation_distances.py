
# Notebook to check representation distances
import glob
import os
import hydra
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import transforms as T
import torch.nn.functional as F

# from agent.encoder import Encoder

from tactile_learning.models import *
from tactile_learning.utils import *
from tactile_learning.tactile_data import *

# Class to analyze data
class RepresentationAnalyzer:
    def __init__(
        self,
        data,
        data_path = '/home/irmak/Workspace/Holo-Bot/extracted_data/bowl_picking/after_rss',
        tactile_out_dir = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.01.28/12-32_tactile_byol_bs_512_tactile_play_data_alexnet_pretrained_duration_120',
        image_out_dir = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.05.06/10-50_image_byol_bs_32_epochs_500_lr_1e-05_bowl_picking_after_rss',
        device = 'cuda',
        reward_representations = ['image','tactile']
    ):
        # Set expert demo
        # roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
        # self.data = load_data(roots, demos_to_use=[expert_demo_num])
        self.data = data
        self.data_path = data_path
        self.device = torch.device(device)

        image_cfg, self.image_encoder, self.image_transform = init_encoder_info(self.device, image_out_dir, 'image')
        self.inv_image_transform = get_inverse_image_norm() 
        self.image_normalize = T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)

        tactile_cfg, self.tactile_encoder, _ = init_encoder_info(self.device, tactile_out_dir, 'tactile', model_type='byol')
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

        self.rewards = 'sinkhorn_cosine'
        self.sinkhorn_rew_scale = 200

        self.reward_representations = reward_representations
        self.policy_representations = ['image', 'tactile', 'features']

        self._set_expert_demo()

    def _load_dataset_image(self, demo_id, image_id):
        dset_img = load_dataset_image(self.data_path, demo_id, image_id, self.view_num)
        img = self.image_transform(dset_img)
        return torch.FloatTensor(img) 

    def _set_expert_demo(self):
        # We'll stack the tactile repr and the image observations
        for step_id in range(len(self.data['image']['indices'])): 
            demo_id, tactile_id = self.data['tactile']['indices'][step_id]

            tactile_value = self.data['tactile']['values'][demo_id][tactile_id]
            tactile_repr = self.tactile_repr.get(tactile_value, detach=False)

            _, image_id = self.data['image']['indices'][step_id]
            image = load_dataset_image(
                data_path = self.data_path, 
                demo_id = demo_id, 
                image_id = image_id,
                view_num = 1,
                transform = self.image_transform
            )

            if step_id == 0:
                tactile_reprs = tactile_repr.unsqueeze(0)
                image_obs = image.unsqueeze(0)
            else:
                image_obs = torch.concat([image_obs, image.unsqueeze(0)], dim=0)
                tactile_reprs = torch.concat([tactile_reprs, tactile_repr.unsqueeze(0)], dim=0)

        self.expert_demo = dict(
            image_obs = image_obs, 
            tactile_repr = tactile_reprs
        )

    def _get_representation_distances(self, episode_obs, mock=False):
        curr_reprs, exp_reprs = [], []
        if 'image' in self.reward_representations: # We will not be using features for reward for sure
            if mock:
                image_reprs = self.image_encoder(episode_obs['image_obs'].to(self.device))
            else: 
                image_obs = self.image_normalize(episode_obs['image_obs']).to(self.device) # This will give all the image observations of one episode
                image_reprs = self.image_encoder(image_obs)
            expert_image_reprs = self.image_encoder(self.expert_demo['image_obs'].to(self.device))
            curr_reprs.append(image_reprs)
            exp_reprs.append(expert_image_reprs)
    
        if 'tactile' in self.reward_representations:
            tactile_reprs = episode_obs['tactile_repr'].to(self.device) # This will give all the representations of one episode
            expert_tactile_reprs = self.expert_demo['tactile_repr'].to(self.device)
            curr_reprs.append(tactile_reprs)
            exp_reprs.append(expert_tactile_reprs)

        # Concatenate everything now
        obs = torch.concat(curr_reprs, dim=-1).detach()
        exp = torch.concat(exp_reprs, dim=-1).detach()

        # Get the rewards
        if self.rewards == 'sinkhorn_cosine':
            cost_matrix = cosine_distance(
                obs, exp)  # Get cost matrix for samples using critic network.
            print('cost_matrix.shape: {}'.format(cost_matrix.shape))
            transport_plan = optimal_transport_plan(
                obs, exp, cost_matrix, method='sinkhorn',
                niter=100).float()  # Getting optimal coupling
            print('ot plan: {}'.format(transport_plan.shape))
            ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
                torch.mm(transport_plan,
                            cost_matrix.T)).detach().cpu().numpy()

        episode_obs = {episode_obs[k].detach().cpu() for k in episode_obs.keys()}
        print('ot_rewards: {}'.format(ot_rewards))

        return ot_rewards

    def ot_rewarder(self, episode_obs, mock=False): # TODO: Delete the mock option
        print('IN OT_REWARDER - torch.cuda.memory_summary: {}'.format(
            torch.cuda.memory_summary(device=self.device)
        ))
        # NOTE: In this code we're not using target encoder since the encoders are already frozen
        curr_reprs, exp_reprs = [], []
        if 'image' in self.reward_representations: # We will not be using features for reward for sure
            if mock:
                image_reprs = self.image_encoder(episode_obs['image_obs'].to(self.device))
            else:
                image_obs = self.image_normalize(episode_obs['image_obs']).to(self.device) # This will give all the image observations of one episode
                image_reprs = self.image_encoder(image_obs)

            print('AFTER IMAGE REPR torch.cuda.memory_summary: {}'.format(
                torch.cuda.memory_summary(device=self.device)
            ))
            print('image_reprs.shape: {}'.format(image_reprs.shape))
            print('SELF.EXPERT_DEMO ID: {}'.format(id(self.expert_demo)))
            # print('self.expert_demo[image_obs].shape: {}'.format(self.expert_demo['image_obs'].shape))
            expert_image_reprs = self.image_encoder(self.expert_demo['image_obs'].to(self.device))
            curr_reprs.append(image_reprs)
            exp_reprs.append(expert_image_reprs)

            # del image_reprs
            # del expert_image_reprs
            # torch.cuda.empty_cache()
    
        if 'tactile' in self.reward_representations:
            tactile_reprs = episode_obs['tactile_repr'].to(self.device) # This will give all the representations of one episode
            expert_tactile_reprs = self.expert_demo['tactile_repr'].to(self.device)
            curr_reprs.append(tactile_reprs)
            exp_reprs.append(expert_tactile_reprs)

            # del tactile_reprs
            # del expert_tactile_reprs
        # torch.cuda.empty_cache()

        # Concatenate everything now
        obs = torch.concat(curr_reprs, dim=-1).detach()
        exp = torch.concat(exp_reprs, dim=-1).detach()

        if self.rewards == 'sinkhorn_cosine':
            cost_matrix = cosine_distance(
                obs, exp)  # Get cost matrix for samples using critic network.
            transport_plan = optimal_transport_plan(
                obs, exp, cost_matrix, method='sinkhorn',
                niter=100).float()  # Getting optimal coupling
            ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
                torch.mm(transport_plan,
                            cost_matrix.T)).detach().cpu().numpy()
            
        elif self.rewards == 'sinkhorn_euclidean':
            cost_matrix = euclidean_distance(
                obs, exp)  # Get cost matrix for samples using critic network.
            transport_plan = optimal_transport_plan(
                obs, exp, cost_matrix, method='sinkhorn',
                niter=100).float()  # Getting optimal coupling
            ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
                torch.mm(transport_plan,
                            cost_matrix.T)).detach().cpu().numpy()
            
        elif self.rewards == 'cosine':
            exp = torch.cat((exp, exp[-1].unsqueeze(0)))
            ot_rewards = -(1. - F.cosine_similarity(obs, exp))
            ot_rewards *= self.sinkhorn_rew_scale
            ot_rewards = ot_rewards.detach().cpu().numpy()
            
        elif self.rewards == 'euclidean':
            exp = torch.cat((exp, exp[-1].unsqueeze(0)))
            ot_rewards = -(obs - exp).norm(dim=1)
            ot_rewards *= self.sinkhorn_rew_scale
            ot_rewards = ot_rewards.detach().cpu().numpy()
            
        else:
            raise NotImplementedError()
        
        # del obs
        # del exp 
        # torch.cuda.empty_cache()

        return ot_rewards

    def _get_reprs_for_reward(self, episode_obs, mock=False):
        curr_reprs, exp_reprs = [], []
        if 'image' in self.reward_representations: # We will not be using features for reward for sure
            if mock:
                image_reprs = self.image_encoder(episode_obs['image_obs'].to(self.device))
            else: 
                image_obs = self.image_normalize(episode_obs['image_obs']).to(self.device) # This will give all the image observations of one episode
                image_reprs = self.image_encoder(image_obs)
            expert_image_reprs = self.image_encoder(self.expert_demo['image_obs'].to(self.device))
            curr_reprs.append(image_reprs)
            exp_reprs.append(expert_image_reprs)
    
        if 'tactile' in self.reward_representations:
            tactile_reprs = episode_obs['tactile_repr'].to(self.device) # This will give all the representations of one episode
            expert_tactile_reprs = self.expert_demo['tactile_repr'].to(self.device)
            curr_reprs.append(tactile_reprs)
            exp_reprs.append(expert_tactile_reprs)

        # Concatenate everything now
        obs = torch.concat(curr_reprs, dim=-1).detach()
        exp = torch.concat(exp_reprs, dim=-1).detach()

        return obs, exp

    def plot_embedding_dists(self, episode_obs, mock=False, file_name=None):
        # Get the observation and expert demonstration torch
        obs, exp = self._get_reprs_for_reward(episode_obs, mock=mock)

        # Get the cost matrix (M,N) -> M: length of the observation, N: length of the expert demo
        cost_matrix = cosine_distance(obs, exp) 
        print('cost_matrix.shape: {}, obs.shape: {}, exp.shape: {}'.format(
            cost_matrix.shape, obs.shape, exp.shape
        ))

        # Plot MxN matrix if file_name is given -> it will save the plot there if so
        if file_name is not None:
            cost_matrix = cost_matrix.detach().cpu().numpy()

            # print('cost_matrix: {}'.format(cost_matrix))
            fig, ax = plt.subplots(figsize=(15,15),nrows=1,ncols=1)
            # im = ax.imshow(data2d)
            im = ax.matshow(cost_matrix)
            ax.set_title(f'File: {file_name}')
            fig.colorbar(im, ax=ax, label='Interactive colorbar')

            plt.show()

            # plt.matshow(cost_matrix)
            plt.savefig(file_name, bbox_inches='tight')
            plt.xlabel('Observation Timesteps')
            plt.ylabel('Expert Demo Timesteps')
            plt.title(file_name)
            plt.close()        

    
if __name__ == '__main__':
    # Get a random episode
    episode_saved_fn = '20230507T204720_14_76'
    fn = f'/home/irmak/Workspace/tactile-learning/buffer/{episode_saved_fn}.npz'
    with open(fn, 'rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}

    data_path = '/home/irmak/Workspace/Holo-Bot/extracted_data/bowl_picking/after_rss' 
    expert_demo_num = 24
    roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
    exp_data = load_data(roots, demos_to_use=[expert_demo_num])
    # mock_data = load_data(roots, demos_to_use=[29]) # When two load_data is used the method tried to put it into GPU?? 

    reward_representations = ['tactile']
    repr_analyzer = RepresentationAnalyzer(
        data = exp_data,
        reward_representations = reward_representations
    )
    episode_obs = dict(
        image_obs = torch.FloatTensor(episode['pixels']),
        tactile_repr = torch.FloatTensor(episode['tactile'])
    )
    repr_analyzer.plot_embedding_dists(
        episode_obs = episode_obs,
        mock = False, 
        file_name = f'rewards_{reward_representations}_{episode_saved_fn}_{expert_demo_num}_dists.png'
    )





    # print('ot_rewards in main: {}'.format(np.sum(ot_rewards)))
    # print('expert demo id before loading data: {}'.format())

    # Get the rewards from an actual demo
    # mock_data = load_data(roots, demos_to_use=[29])
    # mock_data = data
    # We'll stack the tactile repr and the image observations
    # mock_episode_obs = dict(
    #     image_obs = [],
    #     tactile_repr = []
    # )
    # for step_id in range(len(mock_data['image']['indices'])): 
    #     demo_id, tactile_id = mock_data['tactile']['indices'][step_id]

    #     tactile_value = mock_data['tactile']['values'][demo_id][tactile_id]
    #     tactile_repr = repr_analyzer.tactile_repr.get(tactile_value, detach=False).detach().cpu()

    #     _, image_id = mock_data['image']['indices'][step_id]
    #     image = load_dataset_image(
    #         data_path = repr_analyzer.data_path, 
    #         demo_id = demo_id, 
    #         image_id = image_id,
    #         view_num = 1,
    #         transform = repr_analyzer.image_transform
    #     ).detach().cpu()

    #     mock_episode_obs['image_obs'].append(torch.FloatTensor(image))
    #     mock_episode_obs['tactile_repr'].append(torch.FloatTensor(tactile_repr))

    # for obs_type in mock_episode_obs.keys():
    #     mock_episode_obs[obs_type] = torch.stack(mock_episode_obs[obs_type], 0)

    # mock_ot_rewards = repr_analyzer.ot_rewarder(mock_episode_obs, mock=False)
    # print('mock_ot_rewards: {}'.format(np.sum(mock_ot_rewards)))
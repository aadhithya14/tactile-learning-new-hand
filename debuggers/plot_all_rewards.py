# Script to plot all the rewards of all trajectories and the closest
# end frame

import os
import hydra
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image
from torchvision import transforms as T
from PIL import Image
# from agent.encoder import Encoder

from tactile_learning.models import *
from tactile_learning.utils import *
from tactile_learning.tactile_data import *


# This class will plot the last frame of the episodes
# the given best expert id's last frame
# and write the reward given to that
class RewardPlotter:
    def __init__(
        self, 
        episode_frames,
        expert_frames,
        expert_ids,
        rewards,
    ):
        self.episode_frames = episode_frames 
        self.expert_frames = expert_frames 
        self.expert_ids = expert_ids 
        self.rewards = rewards

    def plot_rewards(self, nrows, ncols, figure_name):
        figsize = ((ncols*2)*5, nrows*5) # there will be two images for each column 
        fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols*2)
        fig.suptitle('All rewards for all episodes')

        # Plot the episodes 
        pbar = tqdm(total=nrows*ncols)
        for row in range(nrows):
            for col in range(ncols):
                if col*nrows+row > len(self.episode_frames)-1:
                    break
                # Plot the episodes
                axs[row][col*2].imshow(self.episode_frames[col*nrows+row])
                # Plot the experts 
                axs[row][col*2+1].imshow(self.expert_frames[self.expert_ids[col*nrows+row]])
                # Print the rewards
                axs[row][col*2+1].set_xlabel(f'REWARD: {self.rewards[col*nrows+row]}')
                axs[row][col*2+1].set_title(f'EXPERT ID: {self.expert_ids[col*nrows+row]}')
                pbar.update(1)
                pbar.set_description('row: {}, col*2+1: {}, ccol*nrows+row: {}'.format(
                    row, col*2+1, col*nrows+row
                ))
                print('row: {}, col*2+1: {}, col*nrows+row: {}'.format(
                    row, col*2+1, col*nrows+row
                ))
        pbar.close()
        plt.savefig(figure_name, bbox_inches='tight')

class Rewarder:
    def __init__(
        self,
        image_encoder,
        episode_demos, 
        expert_demos,
        frames_to_match, # Number of frames to match in the end
        match_both, # If true frames to match is used in both the expert and the episode
        # reward_representations=['image']
        device,
    ):
        self.image_encoder = image_encoder
        self.episode_demos = episode_demos # This is 
        self.expert_demos = expert_demos
        self.frames_to_match = frames_to_match
        self.match_both = match_both 
        self.device = device 
        # self.reward_representations = reward_representations
        self.sinkhorn_rew_scale = 200
        self.auto_rew_scale_factor = 10

    # Will get the reward for each episde with the best expert
    # and will return expert ids, and the rewards for each episode
    def get_ot_rewards(self, reward_representations, expo_weight_init=False, return_all_rewards=False):

        best_rewards = [] 
        best_experts = []
        ot_rewards_to_return = []
        for episode_id in range(len(self.episode_demos)):
            all_ot_rewards = []
            best_reward_sum = -sys.maxsize 
            best_ot_reward_id = -1

            for expert_id in range(len(self.expert_demos)):

                obs, exp = self.get_reprs_for_reward(episode_id, expert_id, reward_representations)
                cost_matrix = cosine_distance(
                        obs, exp)  # Get cost matrix for samples using critic network.
                transport_plan = optimal_transport_plan(
                    obs, exp, cost_matrix, method='sinkhorn',
                    niter=100, exponential_weight_init=expo_weight_init).float()  # Getting optimal coupling
                ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
                    torch.mm(transport_plan,
                                cost_matrix.T)).detach().cpu().numpy()
                
                all_ot_rewards.append(ot_rewards)
                sum_ot_rewards = np.sum(ot_rewards) 
                if sum_ot_rewards > best_reward_sum:
                    best_reward_sum = sum_ot_rewards
                    best_ot_reward_id = expert_id

            # Normalize the rewards
            if episode_id == 0:
                self.sinkhorn_rew_scale = self.sinkhorn_rew_scale * self.auto_rew_scale_factor / float(np.abs(best_reward_sum))
                obs, exp = self.get_reprs_for_reward(episode_id, best_ot_reward_id, reward_representations)
                cost_matrix = cosine_distance(
                        obs, exp)  # Get cost matrix for samples using critic network.
                transport_plan = optimal_transport_plan(
                    obs, exp, cost_matrix, method='sinkhorn',
                    niter=100).float()  # Getting optimal coupling
                ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
                    torch.mm(transport_plan,
                                cost_matrix.T)).detach().cpu().numpy()
                
                best_reward_sum = np.sum(ot_rewards)

            # Append them to the best experts and rewards 
            best_rewards.append(best_reward_sum)
            best_experts.append(best_ot_reward_id)

            # Return all of the rewards if asked for
            if return_all_rewards:
                # Get all ot rewards
                ot_rewards_to_return.append(
                    all_ot_rewards[best_ot_reward_id]
                )

        if return_all_rewards:
            return best_rewards, best_experts, ot_rewards_to_return

        return best_rewards, best_experts


    def get_reprs_for_reward(self, episode_id, expert_id, reward_representations):
        curr_reprs = []
        exp_reprs = []
        if 'image' in reward_representations: # We will not be using features for reward for sure
            image_reprs = self.image_encoder(self.episode_demos[episode_id]['image_obs'][-self.frames_to_match:,:].to(self.device))
            expert_image_reprs = self.image_encoder(self.expert_demos[expert_id]['image_obs'][-self.frames_to_match:,:].to(self.device))
            curr_reprs.append(image_reprs)
            exp_reprs.append(expert_image_reprs)
    
        if 'tactile' in reward_representations:
            tactile_reprs = self.episode_demos[episode_id]['tactile_repr'][-self.frames_to_match:,:].to(self.device) # This will give all the representations of one episode
            expert_tactile_reprs = self.expert_demos[expert_id]['tactile_repr'][-self.frames_to_match:,:].to(self.device)
            curr_reprs.append(tactile_reprs)
            exp_reprs.append(expert_tactile_reprs)

        # Concatenate everything now
        obs = torch.concat(curr_reprs, dim=-1).detach()
        exp = torch.concat(exp_reprs, dim=-1).detach()

        return obs, exp
    
def prep_demos_for_visualization(episode_demos, expert_demos, inv_image_transform=None, sort=False, rewards=None):
    if inv_image_transform is None:
        inv_image_transform = get_inverse_image_norm()

    expert_frames = []
    episode_frames = []

    if sort:
        sorted_reward_ids = np.argsort(rewards)
    else:
        sorted_reward_ids = range(len(rewards))

    for episode_id in sorted_reward_ids:
        transformed_img = inv_image_transform(episode_demos[episode_id]['image_obs'][-1])
        episode_frames.append(
            torch.permute(transformed_img, (1,2,0)) # Will only plot the last frame
        )
    
    for expert_id in range(len(expert_demos)):
        transformed_img = inv_image_transform(expert_demos[expert_id]['image_obs'][-1])
        expert_frames.append(
            torch.permute(transformed_img, (1,2,0)) # The last frame onlu
        )

    return expert_frames, episode_frames, sorted_reward_ids

def load_one_episode(fn):
    with open(fn, 'rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
    return episode

def load_root_episodes(root_path):
    episodes = []
    print()
    fns = sorted(glob.glob(f'{root_path}/*.npz'))
    # print('fns: {}'.format(fns))
    for i,fn in enumerate(fns):
        episode = load_one_episode(fn)
        episodes.append(episode)

    return episodes

def load_all_episodes(root_path = None, roots = None):
    if roots is None:
        roots = glob.glob(f'{root_path}/*')
    all_episodes = []
    for root in roots:
        print('loaded root: {}'.format(root))
        root_episodes = load_root_episodes(root)
        all_episodes += root_episodes

    print('len(all_episodes): {}'.format(len(all_episodes)))
    return all_episodes

# This image transform will have the totensor and normalization only
def load_episode_demos(all_episodes, image_transform):
    episode_demos = []
    for episode in all_episodes:
        transformed_image_obs = []
        # tactile_reprs = []
        for image_obs in episode['pixels']:
            pil_image = Image.fromarray(np.transpose(image_obs, (1,2,0)), 'RGB')
            # print('pil_image.shape: {}'.format(pil_image.shape))
            transformed_image_obs.append(
                image_transform(pil_image)
            )  
        episode_demos.append(dict(
            image_obs = torch.stack(transformed_image_obs, 0),
            tactile_repr = torch.FloatTensor(episode['tactile'])
        ))

    return episode_demos

# This image transform will have everything
def load_expert_demos(data_path, expert_demo_nums, tactile_repr_module, image_transform, view_num):
    roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
    data = load_data(roots, demos_to_use=expert_demo_nums)
    
    expert_demos = []
    image_obs = [] 
    tactile_reprs = []
    old_demo_id = -1
    for step_id in range(len(data['image']['indices'])): 
        demo_id, tactile_id = data['tactile']['indices'][step_id]
        if (demo_id != old_demo_id and step_id > 0) or (step_id == len(data['image']['indices'])-1): # NOTE: We are losing the last frame of the last expert

            expert_demos.append(dict(
                image_obs = torch.stack(image_obs, 0), 
                tactile_repr = torch.stack(tactile_reprs, 0)
            ))
            image_obs = [] 
            tactile_reprs = []

        tactile_value = data['tactile']['values'][demo_id][tactile_id]
        tactile_repr = tactile_repr_module.get(tactile_value, detach=False)

        _, image_id = data['image']['indices'][step_id]
        image = load_dataset_image(
            data_path = data_path, 
            demo_id = demo_id, 
            image_id = image_id,
            view_num = view_num,
            transform = image_transform
        )
        image_obs.append(image)
        tactile_reprs.append(tactile_repr)


        old_demo_id = demo_id

    return expert_demos

# @hydra.main(version_base=None, config_path='../tactile_learning/configs', config_name='debug')
def get_all_demos(cfg: DictConfig):
    # cfg = cfg.plot_all_rewards
    device = torch.device(cfg.device)
    tactile_cfg, tactile_encoder, _ = init_encoder_info(device, cfg.tactile_out_dir, 'tactile', model_type='byol')
    tactile_img = TactileImage(
        tactile_image_size = tactile_cfg.tactile_image_size, 
        shuffle_type = None
    )
    tactile_repr_module = TactileRepresentation( # This will be used when calculating the reward - not getting the observations
        encoder_out_dim = tactile_cfg.encoder.out_dim,
        tactile_encoder = tactile_encoder,
        tactile_image = tactile_img,
        representation_type = 'tdex',
        device = device
    )

    def viewed_crop_transform(image):
        return crop_transform(image, camera_view=cfg.view_num)
    expert_demos = load_expert_demos(
        data_path = cfg.data_path,
        expert_demo_nums=cfg.expert_demo_nums,
        tactile_repr_module=tactile_repr_module,
        image_transform = T.Compose([
            T.Resize((480,640)),
            T.Lambda(viewed_crop_transform),
            T.Resize(480),
            T.ToTensor(),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS), 
        ]),
        view_num = cfg.view_num
    )

    all_episodes = load_all_episodes(
        roots = cfg.episode_roots, # Either one of them will be none
        root_path = cfg.episode_root_path
    )
    episode_demos = load_episode_demos(
        all_episodes= all_episodes,
        image_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)
        ])
    )

    return expert_demos, episode_demos

@hydra.main(version_base=None, config_path='../tactile_learning/configs', config_name='debug')
def get_all_rewards(cfg: DictConfig):
    cfg = cfg.plot_all_rewards 

    # Get all episodes first 
    expert_demos, episode_demos = get_all_demos(cfg)
    print(len(expert_demos), len(episode_demos))

    # Get all the rewards
    _, image_encoder, _ = init_encoder_info(
        device = torch.device(cfg.device),
        out_dir = cfg.image_out_dir,
        encoder_type = 'image',
        view_num = cfg.view_num,
        model_type = 'bc'
    )
    rewarder = Rewarder(
        image_encoder = image_encoder,
        episode_demos = episode_demos,
        expert_demos = expert_demos, 
        frames_to_match = cfg.frames_to_match,
        match_both = cfg.match_both,
        device = cfg.device
    )
    best_rewards, best_expert_ids = rewarder.get_ot_rewards(
        reward_representations = cfg.reward_representations,
        expo_weight_init = cfg.expo_weight_init
    )
    print('best_rewards: {}, best_expert_ids: {}'.format(best_rewards, best_expert_ids))


    # Sort and plot them
    expert_frames, episode_frames, sorted_reward_ids = prep_demos_for_visualization(
        expert_demos=expert_demos,
        episode_demos=episode_demos,
        sort=True,
        rewards=best_rewards
    )
    print('len(expert_frames): {}, len(episode_Frames): {}'.format(
        len(expert_frames), 
        len(episode_frames)
    ))

    print('sorted_reward_ids: {}'.format(sorted_reward_ids))

    sorted_expert_ids = [best_expert_ids[reward_id] for reward_id in sorted_reward_ids]
    sorted_rewards = [best_rewards[reward_id] for reward_id in sorted_reward_ids]
    plotter = RewardPlotter(
        episode_frames = episode_frames,
        expert_frames = expert_frames, 
        expert_ids = sorted_expert_ids,
        rewards = sorted_rewards
    ) 
    ncols = 1
    # nrows = min(int(len(episode_frames)/ncols), 60)
    nrows = int(len(episode_frames)/ncols)
    print('ncols: {}, nrows: {}'.format(ncols, nrows))
    episode_time_step = cfg.episode_roots[0].split('/')[-1].split('_')[0]
    # Remove the dots from the string
    ts_arr = episode_time_step.split('.')
    episode_ts = ''.join(ts_arr)
    print('episode_ts: {}'.format(episode_ts))
    plotter.plot_rewards(
        nrows = nrows,
        ncols = ncols,
        figure_name = f'plot_all_rewards_outputs/{cfg.object}_{episode_ts}_experts_{cfg.expert_demo_nums}_match_frames_{cfg.frames_to_match}_mb_{cfg.match_both}_rewards_{cfg.reward_representations}_expo_{cfg.expo_weight_init}_sorted'
    )

if __name__ == '__main__':
    get_all_rewards()
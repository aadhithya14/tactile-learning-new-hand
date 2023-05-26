# Given a buffer root - modify the reward at each timestep
# at that buffer with the given rewards

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
from tactile_learning.datasets import *

from plot_all_rewards import *

def save_one_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with open(fn, 'wb') as f:
            f.write(bs.read())

def update_episode_reward(episode_reward, fn):
    # Given the function name we'll update the rewards
    episode = load_one_episode(fn)

    print('episode reward before change: {}'.format(
        episode['reward']
    ))
    # Traverse through the reward and change it
    for reward_id, reward in enumerate(episode['reward']):
        new_reward = episode_reward[reward_id]
        episode['reward'][reward_id] = new_reward

    print('episode reward after change: {}'.format(
        episode['reward'] 
    ))

    print('fn: {}'.format(fn))

    # Save the episode
    # save_one_episode(episode, fn)


@hydra.main(version_base=None, config_path='../tactile_learning/configs', config_name='debug')
def change_all_rewards(cfg: DictConfig):
    cfg = cfg.reward_changer
    
    # Get the expert and episde_demos
    expert_demos, episode_demos = get_all_demos(cfg)

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
    _, _, best_rewards = rewarder.get_ot_rewards(
        reward_representations = cfg.reward_representations,
        expo_weight_init = cfg.expo_weight_init,
        return_all_rewards = True
    )

    # print('best_rewards: {}'.format(best_reward_sums))

    episode_fns = []
    for root in cfg.episode_roots:
        fns = sorted(glob.glob(f'{root}/*.npz'))
        episode_fns += fns

    assert len(episode_fns) == len(best_rewards)

    # Now update each episode
    for episode_id in range(len(episode_fns)):
        episode_fn = episode_fns[episode_id]
        episode_reward = best_rewards[episode_id]

        print(f'episode id: {episode_id}')

        update_episode_reward(
            episode_reward = episode_reward, 
            fn = episode_fn
        )

        print('-----')


if __name__ == '__main__': 
    change_all_rewards()
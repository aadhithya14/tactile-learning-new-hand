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
from tactile_learning.utils import load_dataset_image
from tactile_learning.tactile_data import *

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

def get_all_demos(cfg):
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
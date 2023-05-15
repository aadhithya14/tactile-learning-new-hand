
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

from PIL import Image
# from agent.encoder import Encoder

from tactile_learning.models import *
from tactile_learning.utils import *
from tactile_learning.tactile_data import *

# Class to analyze data
class RepresentationAnalyzer:
    def __init__(
        self,
        data,
        cfg,
        data_path = '/home/irmak/Workspace/Holo-Bot/extracted_data/bowl_picking/after_rss',
        tactile_out_dir = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.01.28/12-32_tactile_byol_bs_512_tactile_play_data_alexnet_pretrained_duration_120',
        # image_out_dir = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.05.06/10-50_image_byol_bs_32_epochs_500_lr_1e-05_bowl_picking_after_rss',
        device = 'cpu',
        view_num = 1
    ):
        # Set expert demo
        # roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
        # self.data = load_data(roots, demos_to_use=[expert_demo_num])
        self.data = data
        self.cfg = cfg
        self.view_num = 1
        self.data_path = data_path
        self.device = torch.device(device)

        # image_cfg, self.image_encoder, self.image_transform = init_encoder_info(self.device, image_out_dir, 'image')
        # self.inv_image_transform = get_inverse_image_norm() 
        self._get_image_encoders()
        self._set_image_transform()
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
            representation_type = 'tdex',
            device = device
        )

        self.rewards = 'sinkhorn_cosine'
        self.sinkhorn_rew_scale = 200

        self._set_expert_demos()

    def _set_image_transform(self):
        def viewed_crop_transform(image):
            return crop_transform(image, camera_view=self.view_num)
        
        self.image_transform = T.Compose([
            T.Resize((480,640)),
            T.Lambda(viewed_crop_transform),
            T.Resize(480),
            T.ToTensor(),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS), 
        ]) 

    def _get_image_encoders(self):
        self.image_encoders = []

        # First load the pretrained targets
        for enc_target in self.cfg.encoder_targets:
            if enc_target.type == 'pretrained':
                encoder = hydra.utils.instantiate(
                    enc_target.encoder,
                    pretrained = self.cfg.params.pretrained, 
                    out_dim = self.cfg.params.out_dim, 
                    remove_last_layer = self.cfg.params.remove_last_layer
                )
            elif enc_target.type == 'trained':
                _, encoder, _ = init_encoder_info(
                    device = self.device,
                    out_dir = enc_target.out_dir,
                    encoder_type = 'image',
                    view_num = self.view_num, 
                    model_type = 'bc'
                ) 
            encoder.eval()
            encoder.to(self.device)
            self.image_encoders.append(encoder)

    def _set_expert_demos(self):
        # We'll stack the tactile repr and the image observations
        self.expert_demos = []
        image_obs = [] 
        tactile_reprs = []
        old_demo_id = -1
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
            image_obs.append(image)
            tactile_reprs.append(tactile_repr)

            if (demo_id != old_demo_id and step_id > 0) or (step_id == len(self.data['image']['indices'])-1):

                self.expert_demos.append(dict(
                    image_obs = torch.stack(image_obs, 0), 
                    tactile_repr = torch.stack(tactile_reprs, 0)
                ))
                image_obs = [] 
                tactile_reprs = []

            old_demo_id = demo_id

    def _get_reprs_for_reward(self, reward_representations, episode_obs, encoder_id, expert_id):
        curr_reprs, exp_reprs = [], []

        # print('self.expert_demos[expert_id][image_obs].shape: {}, episode_obs[image_obs].shape: {}'.format(
        #     self.expert_demos[expert_id]['image_obs'].shape, episode_obs['image_obs'].shape
        # ))

        if 'image' in reward_representations: # We will not be using features for reward for sure
            image_reprs = self.image_encoders[encoder_id](episode_obs['image_obs'].to(self.device))
            expert_image_reprs = self.image_encoders[encoder_id](self.expert_demos[expert_id]['image_obs'].to(self.device))
            curr_reprs.append(image_reprs)
            exp_reprs.append(expert_image_reprs)
    
        if 'tactile' in reward_representations:
            tactile_reprs = episode_obs['tactile_repr'].to(self.device) # This will give all the representations of one episode
            expert_tactile_reprs = self.expert_demos[expert_id]['tactile_repr'].to(self.device)
            curr_reprs.append(tactile_reprs)
            exp_reprs.append(expert_tactile_reprs)

        # Concatenate everything now
        obs = torch.concat(curr_reprs, dim=-1).detach()
        exp = torch.concat(exp_reprs, dim=-1).detach()

        return obs, exp

    def plot_embedding_dists(self, reward_representations, episode_obs, encoder_id, expert_id, file_name=None):
        # Get the observation and expert demonstration torch
        obs, exp = self._get_reprs_for_reward(reward_representations, episode_obs, encoder_id, expert_id)

        # Get the cost matrix (M,N) -> M: length of the observation, N: length of the expert demo
        cost_matrix = cosine_distance(obs, exp) 

        # Plot MxN matrix if file_name is given -> it will save the plot there if so
        if file_name is not None:
            cost_matrix = cost_matrix.detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(15,15),nrows=1,ncols=1)
            im = ax.matshow(cost_matrix)
            ax.set_title(f'File: {file_name}')
            fig.colorbar(im, ax=ax, label='Interactive colorbar')

            # plt.matshow(cost_matrix)
            plt.savefig(file_name, bbox_inches='tight')
            plt.xlabel('Observation Timesteps')
            plt.ylabel('Expert Demo Timesteps')
            plt.title(file_name)
            plt.close()        

@hydra.main(version_base=None, config_path='../tactile_learning/configs', config_name='debug')
def main(cfg: DictConfig) -> None:

    if not cfg.mock:
        # Get a random episode
        episode_saved_fn = cfg.episode_saved_fn
        fn = f'/home/irmak/Workspace/tactile-learning/buffer/{episode_saved_fn}.npz'
        with open(fn, 'rb') as f:
            episode = np.load(f)
            episode = {k: episode[k] for k in episode.keys()}
    else:
        episode_saved_fn = f'expert_{cfg.mock_traj_id}'

    # Load the expert data
    data_path = '/home/irmak/Workspace/Holo-Bot/extracted_data/bowl_picking/after_rss' 
    roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
    print('roots: {}'.format(roots))
    exp_data = load_data(roots, demos_to_use=[])

    # Initialize the representation analyzer
    repr_analyzer = RepresentationAnalyzer(
        data = exp_data,
        cfg = cfg
    )

    if not cfg.mock:
        # Turn the episode observations to similar to what we have in expert demonstrations
        pil_image_obs = torch.zeros((episode['pixels'].shape[0], 3, 480, 480))
        for i in range(episode['pixels'].shape[0]):
            
            pil_image_obs[i,:] = repr_analyzer.image_transform(
                Image.fromarray(np.transpose(episode['pixels'][i,:], (1,2,0)), 'RGB')
            )

        episode_obs = dict(
            image_obs = pil_image_obs,
            tactile_repr = torch.FloatTensor(episode['tactile'])
        )
    else:
        episode_obs = dict(
            image_obs = repr_analyzer.expert_demos[cfg.mock_traj_id]['image_obs'],
            tactile_repr = repr_analyzer.expert_demos[cfg.mock_traj_id]['tactile_repr']
        )

    # Plot encoder configs
    for i,enc_target in enumerate(cfg.encoder_targets):
        print('encoder_id: {}, encoder name: {}'.format(
            i, enc_target.name
        ))

    # Plot the cosine matrices
    pbar = tqdm(total = len(cfg.encoder_targets) * 6)
    for encoder_id, encoder_target in enumerate(cfg.encoder_targets):
        for reward_representations in [['image'], ['image', 'tactile']]:
            for expert_id in range(6):
                file_name = f'all_encoder_outputs/encoder_{encoder_target.name}_rewards_{reward_representations}_expert_{expert_id}_{episode_saved_fn}_dists.png'
                repr_analyzer.plot_embedding_dists(
                    reward_representations = reward_representations,
                    episode_obs = episode_obs,
                    encoder_id = encoder_id,
                    expert_id = expert_id,
                    file_name = file_name
                )
                pbar.update(1) 
                pbar.set_description(f'Dumped: {file_name}')
                

if __name__ == '__main__':
    main()
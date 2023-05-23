# Script to randomly get frames from the episodes that are saved
# and plot the distances bw the representations for each frame

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

from numpy import dot
from numpy.linalg import norm

from PIL import Image
# from agent.encoder import Encoder

from tactile_learning.models import *
from tactile_learning.utils import *
from tactile_learning.tactile_data import *

class EpisodeDistanceCheck:
    def __init__(
        self, 
        data_path,
        tactile_repr, # Tactile representation module
        image_encoder,
        device,
        buffer_roots = None, # List of buffer roots to load the npz files from
        expert_w_expert = False,
        single_episode_demo = True,
        episode_fn = None, 
        expert_demo_nums = [],
        test_demo_nums = [], 
        distance_type = 'cosine', # It could be L2 norm as well - if that makes more sense!! 
        view_num = 1,
        reward_representations = ['image', 'tactile']
    ):
        # Load the whole data - will need to get all representations
        self.data_path = data_path 
        self.view_num = view_num
        self.distance_type = distance_type

        roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
        self.exp_data = load_data(roots, demos_to_use=expert_demo_nums)
        if expert_w_expert:
            assert test_demo_nums != expert_demo_nums # NOTE: Check this...
            self.test_data = load_data(roots, demos_to_use=test_demo_nums)

        # Load the episodes
        self.episodes = []
        if single_episode_demo:
            with open(episode_fn, 'rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
            self.episodes.append(episode)
        else:
            for buffer_root in buffer_roots:
                fns = glob.glob(f'{buffer_root}/*.npz')
                for i,fn in enumerate(fns):
                    with open(fn, 'rb') as f:
                        episode = np.load(f)
                        episode = {k: episode[k] for k in episode.keys()}
                    # print('fn: {}, episode.keys(): {}'.format(fn, episode.keys()))
                    # print(f'episode {i}, fn: {fn}, episode shapes: image={episode["pixels"].shape}, tactile={episode["tactile"].shape}')
                    self.episodes.append(episode)

        # Set the tactile repr and imag eencoder 
        self.tactile_repr = tactile_repr
        self.image_encoder = image_encoder 
        self.device = device 
        self.single_episode_demo = single_episode_demo

        # Get all the representations
        self._set_image_transform()

        # print('self.episodes: {}'.format(self.episodes))

        self.all_exp_representations = self._get_all_exp_representations(data = self.exp_data, reward_representations=reward_representations) 
        print('self.all_exp_repr.shape: {}'.format(self.all_exp_representations.shape))
        if expert_w_expert:
            self.all_test_exp_representations = self._get_all_exp_representations(data = self.test_data, reward_representations = reward_representations)
            print('self.all_test_exp_repr.shape: {}'.format(self.all_test_exp_representations.shape))
        else:
            self._get_all_episode_representations(reward_representations=reward_representations)

    def _get_random_episodes(self, N):
        random_ids = np.random.choice(range(self.all_episode_representations.shape[0]), size=N)
        return self.all_episode_representations[random_ids,:], random_ids
    
    # We assume that this function is only called with one single episode 
    def _get_ordered_episode_reprs(self, N): 
        # Get the id distance bw each representation to plot
        id_dist = int(self.all_episode_representations.shape[0] / N)
        print(self.all_episode_representations.shape[0], N, id_dist)

        repr_ids = np.array([repr_id for repr_id in range(0, self.all_episode_representations.shape[0], id_dist)], dtype=np.uint)

        print(f'repr_ids.shape: {repr_ids.shape}')

        return self.all_episode_representations[repr_ids[-N:],:], repr_ids[-N:] 

    def _get_closest_exp_observations(self, episode_repr, k=5):
        if self.distance_type == 'cosine':
            cost_matrix = cosine_distance(
                torch.FloatTensor(episode_repr),
                torch.FloatTensor(self.all_exp_representations)
            )
        elif self.distance_type == 'euclidean':
            cost_matrix = euclidean_distance(
                torch.FloatTensor(episode_repr),
                torch.FloatTensor(self.all_exp_representations)
            )
        elif self.distance_type == 'l2_norm':
            l1_distances = torch.FloatTensor(self.all_exp_representations) - torch.FloatTensor(episode_repr)
            cost_matrix = torch.linalg.norm(l1_distances, axis=1)
        print('self.distance_type: {}, cost_matrix.shape: {}'.format(self.distance_type, cost_matrix.shape)) # It should be (1, N) N-> Number of expert representations

        # Squeeze and argsort the cost matrix
        cost_matrix = cost_matrix.squeeze()
        # print('squeezed cost_matrix.shape: {}'.format(cost_matrix.shape))

        # Argsort
        sorted_exp_ids = torch.argsort(cost_matrix, dim=-1, descending=False)[:k]

        return cost_matrix[sorted_exp_ids], sorted_exp_ids

    # Plots episode representations with expert neighbors
    def plot_close_representations(self, N, k, figure_name): # N: Number of tests 

        # Create the plot
        figsize = ((k+1)*4, N*4) # k+1 columns, N rows
        fig, axs = plt.subplots(figsize=figsize, nrows=N, ncols=k+1)
        fig.suptitle(f'Neighbors for random episode observations')
        axs[0][0].set_title("Actual")
        for i in range(k):
            axs[0][i+1].set_title(f'{i+1}th Neighbor')

        # Get random episode representatoins
        if self.single_episode_demo:
            episode_reprs, episode_ids = self._get_ordered_episode_reprs(N)
        else:
            episode_reprs, episode_ids = self._get_random_episodes(N)

        # Plot the actual observations for the first column
        for i in range(N):
            # Get the image observation with that episode id
            curr_episode_id = episode_ids[i]
            episode_img = self._get_episode_image_with_id(curr_episode_id)

            axs[i][0].imshow(episode_img)

        # Traverse through each and get the closest expert representations
        for i, episode_repr in enumerate(episode_reprs):
            # print('episode_repr.shape: {}'.format(episode_repr.shape)) # Should be 1536

            closest_exp_dists, closest_exp_ids = self._get_closest_exp_observations(np.expand_dims(episode_repr, 0), k)

            # Plot the images for the neighbors
            for j,exp_id in enumerate(closest_exp_ids):
                exp_img = self._get_data_with_id(data=self.exp_data, id=exp_id, visualize=True)['image']
                axs[i][j+1].imshow(exp_img)
                axs[i][j+1].set_xlabel('Dists: {}'.format(closest_exp_dists[j]))


        plt.savefig(figure_name, bbox_inches='tight')

    def _get_ordered_test_exp_representations(self, N):
        # Get the id distance bw each representation to plot
        id_dist = int(self.all_test_exp_representations.shape[0] / N)
        print(self.all_test_exp_representations.shape[0], N, id_dist)

        repr_ids = np.array([repr_id for repr_id in range(0, self.all_test_exp_representations.shape[0], id_dist)], dtype=np.uint)

        print(f'repr_ids.shape: {repr_ids.shape}')

        return self.all_test_exp_representations[repr_ids[-N:],:], repr_ids[-N:]

    # Plots expert reprs with expert neighbors 
    def plot_close_exp_representations(self, N, k, figure_name):

        # Create the plot
        figsize = ((k+1)*4, N*4) # k+1 columns, N rows
        fig, axs = plt.subplots(figsize=figsize, nrows=N, ncols=k+1)
        fig.suptitle(f'Neighbors for random episode observations')
        axs[0][0].set_title("Actual")
        for i in range(k):
            axs[0][i+1].set_title(f'{i+1}th Neighbor')

        # Get random episode representatoins
        test_reprs, test_ids = self._get_ordered_test_exp_representations(N)

        # Plot the actual observations for the first column
        for i in range(N):
            # Get the image observation with that episode id
            curr_test_id = test_ids[i]
            test_img= self._get_data_with_id(data=self.test_data, id=curr_test_id, visualize=True)['image'] 

            axs[i][0].imshow(test_img)

        # Traverse through each and get the closest expert representations
        # i = 0
        for i,test_repr in enumerate(test_reprs):
            closest_exp_dists, closest_exp_ids = self._get_closest_exp_observations(np.expand_dims(test_repr, 0), k)

            # Plot the images for the neighbors
            for j,exp_id in enumerate(closest_exp_ids):
                exp_img = self._get_data_with_id(data=self.exp_data, id=exp_id, visualize=True)['image']
                axs[i][j+1].imshow(exp_img)
                axs[i][j+1].set_xlabel('Dists: {}'.format(closest_exp_dists[j]))

            # i += 1

        plt.savefig(figure_name, bbox_inches='tight')


    def _set_image_transform(self):
        def viewed_crop_transform(image):
            return crop_transform(image, camera_view=self.view_num)
            
        self.image_exp_transform = T.Compose([
            T.Resize((480,640)),
            T.Lambda(viewed_crop_transform),
            T.Resize(480),
            T.ToTensor(),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS),
        ]) 
        self.image_episode_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)
        ])

        self.image_visualization_transform = T.Compose([
            T.Resize((480,640)),
            T.Lambda(viewed_crop_transform),
            T.Resize(480)
        ])

    # tactile_values: (N,16,3) - N: Number of sensors
    # robot_states: { allegro: allegro_tip_positions: 12 - 3*4, End effector cartesian position for each finger tip
    #                 kinova: kinova_states : (3,) - Cartesian position of the arm end effector}
    def _get_one_representation(self, image, tactile_repr, reward_representations=['image', 'tactile']):
        curr_repr = []
        for i,repr_type in enumerate(reward_representations):
            if repr_type == 'tactile':
                # tactile_values = torch.FloatTensor(tactile_values).to(self.device)
                # new_repr = self.tactile_repr.get(tactile_values)
                curr_repr.append(tactile_repr.detach().cpu().numpy()) # NOTE: This might not work due to tactile_repr difference
                # print('tactile_repr.shape: {}'.format(tactile_repr.shape))
            elif repr_type == 'image':
                new_repr = self.image_encoder(image.unsqueeze(dim=0).to(self.device)) # Add a dimension to the first axis so that it could be considered as a batch
                new_repr = new_repr.detach().cpu().numpy().squeeze()
                curr_repr.append(new_repr)
                # print('image_repr.shape: {}'.format(new_repr.shape))
        
        return np.concatenate(curr_repr, axis=-1)
    
    def _get_episode_image_with_id(self, id):
        i = 0
        for episode in self.episodes:
            for image_obs in episode['pixels']:
                if i == id:
                    return Image.fromarray(np.transpose(image_obs, (1,2,0)), 'RGB')
                i += 1

        return None
    
    def _get_all_episode_representations(self, reward_representations=['image', 'tactile']):
        print('Getting all episode representations')
        all_image_reprs = []
        all_tactile_reprs = []
        pbar = tqdm(total=len(self.episodes))
        for episode in self.episodes: 
            # Get the episode image representations 
            # Pass each image in the episode separately through the transform and create the reprs like that 
            transformed_image_obs = []
            for image_obs in episode['pixels']:
                pil_image = Image.fromarray(np.transpose(image_obs, (1,2,0)), 'RGB')
                transformed_image_obs.append(
                    self.image_episode_transform(pil_image).to(self.device)
                )

            image_repr = torch.stack(transformed_image_obs, dim=0) # Concatenate all of them afterwards 
            image_repr = self.image_encoder(image_repr).detach().cpu().numpy()
            all_image_reprs.append(image_repr)

            # Get the episode tactile representation
            tactile_repr = episode['tactile']
            all_tactile_reprs.append(tactile_repr)

            pbar.update(1)

        pbar.close()

        image_reprs = np.concatenate(all_image_reprs, axis=0)
        tactile_reprs = np.concatenate(all_tactile_reprs, axis=0)


        reprs = []
        if 'tactile' in reward_representations: reprs.append(tactile_reprs)
        if 'image' in reward_representations: reprs.append(image_reprs)

        self.all_episode_representations = np.concatenate(reprs, axis=-1)

        print('all_episode_reprs.shape: {}'.format(self.all_episode_representations.shape))
    
    def _get_all_exp_representations(self, data, reward_representations=['image', 'tactile']):
        print('Getting all expert representations')
        all_reprs = []
        pbar = tqdm(total=len(data['tactile']['indices']))
        for index in range(len(data['tactile']['indices'])):
            # Get the representation data
            repr_data = self._get_data_with_id(data, index, visualize=False)

            representation = self._get_one_representation(
                repr_data['image'],
                repr_data['tactile_repr'],
                reward_representations=reward_representations
            )
            # self.all_representations[index, :] = representation[:]
            all_reprs.append(representation)
            pbar.update(1)

        pbar.close()

        # self.all_exp_representations = np.stack(all_reprs, axis=0)

        # print('all_exp_representations.shape: {}'.format(self.all_exp_representations.shape))

        return np.stack(all_reprs, axis=0)

    def _get_data_with_id(self, data, id, visualize=False):
        demo_id, tactile_id = data['tactile']['indices'][id]
        _, image_id = data['image']['indices'][id]

        tactile_value = data['tactile']['values'][demo_id][tactile_id] # This should be (N,16,3)
        tactile_repr = self.tactile_repr.get(tactile_value, detach=False)
        image_transform = self.image_visualization_transform if visualize else self.image_exp_transform # Don't transform if we want to visualize
        image = load_dataset_image(
            data_path = self.data_path,
            demo_id = demo_id, 
            image_id = image_id,
            view_num = self.view_num,
            transform = image_transform
        )

        return dict(
            image = image,
            tactile_repr = tactile_repr, 
        )


if __name__ == '__main__': 
    data_path = '/home/irmak/Workspace/Holo-Bot/extracted_data/bowl_picking/after_rss'
    device_name = 'cuda:0'
    device = torch.device(device_name)
    
    # Get the tactile representation module
    tactile_out_dir = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.01.28/12-32_tactile_byol_bs_512_tactile_play_data_alexnet_pretrained_duration_120'
    tactile_cfg, tactile_encoder, _ = init_encoder_info(device, tactile_out_dir, 'tactile', model_type='byol')
    tactile_img = TactileImage(
        tactile_image_size = tactile_cfg.tactile_image_size, 
        shuffle_type = None
    )
    tactile_repr = TactileRepresentation( # This will be used when calculating the reward - not getting the observations
        encoder_out_dim = tactile_cfg.encoder.out_dim,
        tactile_encoder = tactile_encoder,
        tactile_image = tactile_img,
        representation_type = 'tdex',
        device = device
    )

    # Get the image encoder
    _, image_encoder, _ = init_encoder_info(
        device = device,
        out_dir = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.05.11/13-21_bc_bs_32_epochs_500_lr_1e-05_bowl_picking_after_rss',
        encoder_type = 'image',
        view_num = 1, 
        model_type = 'bc'
    ) 
    image_encoder.eval()
    image_encoder.to(device)

    # Set the buffer roots
    buffer_roots = [
        # '/home/irmak/Workspace/tactile-learning/buffer/2023.05.15T13-53_costs_summed',
        # '/home/irmak/Workspace/tactile-learning/buffer/2023.05.15T16-16_end_frames_stacked_only_arm',
        # '/home/irmak/Workspace/tactile-learning/buffer/20230515T11_two_fingers_and_arm_image_tactile_both_last_scene_match'
    ]

    # Single episode check
    single_episode = True
    episode_fn_path = '/home/irmak/Workspace/tactile-learning/buffer/2023.05.19T16-28_last_frames_1_se_False_endrepeat_1_one_expert_both_last_frame/20230519T162854_0_76.npz'

    # Start the representations 
    distance_type = 'cosine'
    repr_type = ['image']
    expert_w_expert = False
    ep_dist_check = EpisodeDistanceCheck(
        data_path = data_path, 
        tactile_repr = tactile_repr,
        image_encoder = image_encoder,
        device = device,
        buffer_roots = buffer_roots,
        expert_w_expert = expert_w_expert, 
        single_episode_demo = single_episode, 
        episode_fn = episode_fn_path,
        expert_demo_nums = [22, 24,26,28,29,34],
        # test_demo_nums = [22], # When we check experts with experts
        distance_type = distance_type,
        view_num = 1,
        reward_representations = repr_type
    ) 

    ep_dist_check.plot_close_representations(N = 20, k = 5, figure_name=f'episode_distance_check_outs/{repr_type[0]}/successful_episode_ewe_{expert_w_expert}_single_episode_{single_episode}_{distance_type}_episode_neighbors.png')
    # ep_dist_check.plot_close_exp_representations(N = 20, k = 10, figure_name = f'episode_distance_check_outs/{repr_type[0]}/ewe_{expert_w_expert}_{distance_type}_episode_neighbors.png')

    # import ipdb; ipdb.set_trace()
    

        
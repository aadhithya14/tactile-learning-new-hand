# Way to test our encoders - using a script rather than a notebook so that we wouldn't
# get out of memory

# Will receive:
# a list of encoders to try for each task
# a list of experts to try the encoders on
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

def calc_traj_score(traj1, traj2):
    # traj1.shape: (80, 512), traj2.shape: (80,512)
    cost_matrix = cosine_distance(
            traj1, traj2)  # Get cost matrix for samples using critic network.
    transport_plan = optimal_transport_plan(
        traj1, traj2, cost_matrix, method='sinkhorn',
        niter=100, exponential_weight_init=False).float().detach().cpu().numpy()

    max_transport_plan = np.max(transport_plan, axis=1) # We are going to find the maximums for traj1
    # print('max_transport_plan.shape: {}, traj1.shape: {}, traj2.shape: {}'.format(
    #     max_transport_plan.shape, traj1.shape, traj2.shape
    # ))
    return np.sum(max_transport_plan)

def get_expert_representations_per_encoder(encoder, task_expert_demos, device):
    # Traverse through all the experts and get the representations
    task_representations = []
    for expert_id in range(len(task_expert_demos)):
        # import ipdb; ipdb.set_trace()
        expert_representations = []
        task_len = len(task_expert_demos[expert_id]['image_obs'])
        for batch_id in range(0, task_len, 10): # in order to prevent cuda out of memory error we load the demos in batches
            batch_repr = encoder(task_expert_demos[expert_id]['image_obs'][batch_id:min(batch_id+10, task_len),:].to(device)).detach().cpu()
            print('batch_repr.shape: {}'.format(batch_repr.shape))
            expert_representations.append(batch_repr)
        # expert_representations = encoder(task_expert_demos[expert_id]['image_obs'].to(device)) # One trajectory representations
        expert_representations = torch.concat(expert_representations, 0)
        print('expert_representations.shape: {}'.format(expert_representations.shape))
        task_representations.append(expert_representations)
    
    return task_representations

def calc_encoder_score(encoder, all_expert_demos, encoder_id, device): # Will get all the representations and calculate the score of the 

    all_expert_representations = get_expert_representations_per_encoder(
        encoder = encoder,
        task_expert_demos = all_expert_demos,
        device = device
    )

    # Get combinations of the trajectories and calculate the score for them
    score_matrix = np.zeros((5,5))
    for i in range(score_matrix.shape[0]):
        for j in range(score_matrix.shape[1]):
            traj1 = all_expert_representations[i] 
            traj2 = all_expert_representations[j] 
            score_matrix[i,j] = calc_traj_score(traj1, traj2)

    print('SCORE MATRIX FOR ENCODER: {} \n{}\n-----'.format(
        encoder_id, 
        score_matrix
    ))

    return score_matrix




# This image transform will have everything
def load_expert_demos_per_task(task_name, expert_demo_nums, view_num):
    data_path = f'/home/irmak/Workspace/Holo-Bot/extracted_data/{task_name}'
    roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
    data = load_data(roots, demos_to_use=expert_demo_nums) # NOTE: This could be fucked up

    # Get the tactile module and the image transform
    def viewed_crop_transform(image):
        return crop_transform(image, camera_view=view_num)
    image_transform =  T.Compose([
        T.Resize((480,640)),
        T.Lambda(viewed_crop_transform),
        T.Resize(480),
        T.ToTensor(),
        T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS), 
    ])
    
    expert_demos = []
    image_obs = [] 
    old_demo_id = -1
    for step_id in range(len(data['image']['indices'])): 
        demo_id, image_id = data['image']['indices'][step_id]
        if (demo_id != old_demo_id and step_id > 0) or (step_id == len(data['image']['indices'])-1): # NOTE: We are losing the last frame of the last expert

            expert_demos.append(dict(
                image_obs = torch.stack(image_obs, 0), 
            ))
            image_obs = [] 

        image = load_dataset_image(
            data_path = data_path, 
            demo_id = demo_id, 
            image_id = image_id,
            view_num = view_num,
            transform = image_transform
        )
        image_obs.append(image)
        # tactile_reprs.append(tactile_repr)


        old_demo_id = demo_id

    return expert_demos


def load_encoder(view_num, model_type, model_path, encoder_fn, device):
    if model_type == 'pretrained' and not (encoder_fn is None):
        # It means that this is pretrained
        image_encoder = encoder_fn(pretrained=True, out_dim=512, remove_last_layer=True).to(device)

    else:
        _, image_encoder, _ = init_encoder_info(
            device = device,
            out_dir = model_path,
            encoder_type = 'image',
            view_num = view_num,
            model_type = model_type
        )

    return image_encoder

@hydra.main(version_base=None, config_path='../tactile_learning/configs', config_name='debug')
def get_encoder_score(cfg: DictConfig): # This will be used to set the model path and etc
    cfg = cfg.encoder_metric_calculator
    task_info = dict(
        encoders = [dict(
            model_path = cfg.model_path,
            model_type = cfg.model_type,
            view_num = cfg.view_num,
            encoder_fn = cfg.encoder_fn,
            device = cfg.device
        )],
        demo = dict(
            task_name = cfg.task_name,
            expert_demo_nums = cfg.expert_demo_nums,
            view_num = cfg.view_num
        )
    )

    encoders = [
        load_encoder(**encoder_args) for encoder_args in task_info['encoders']
    ]
    print('LOADED THE ENCODERS')
    # bowl_unstacking_demos = [load_expert_demos_per_task(**bowl_unstacking_info['demos']) for i in range(4)]
    demos = load_expert_demos_per_task(**task_info['demos'])
    print('LOADED THE DEMOS')

    for encoder_id, encoder in enumerate(encoders):
        score_matrix = calc_encoder_score(
            encoder = encoder,
            all_expert_demos = demos, 
            encoder_id = encoder_id,
            device = encoder_id
        )
        encoder_score = np.sum(score_matrix)

        print('id: {} encoder_score: {}'.format(encoder_id, encoder_score))
        print('----')
        _ = input("Press Enter to continue... ")


if __name__ == '__main__':
    # First let's try
    get_encoder_score()


    # bowl_unstacking_info = dict(
    #     encoders = [
    #         dict(
    #             model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.05.11/13-21_bc_bs_32_epochs_500_lr_1e-05_bowl_picking_after_rss',
    #             model_type = 'bc',
    #             view_num = 1,
    #             encoder_fn = None,
    #             device = 0
    #         ),
    #         dict(
    #             model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.06/18-27_temporal_ssl_bs_32_epochs_1000_view_1_bowl_picking_frame_diff_5_resnet',
    #             model_type = 'temporal',
    #             view_num = 1,
    #             encoder_fn = None,
    #             device = 1,
    #         ),
    #         dict(
    #             model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.05.06/10-50_image_byol_bs_32_epochs_500_lr_1e-05_bowl_picking_after_rss',
    #             model_type = 'byol',
    #             view_num = 1,
    #             encoder_fn = None,
    #             device = 2, 
    #         ),
    #         dict(
    #             model_path = None,
    #             model_type = 'pretrained',
    #             encoder_fn = resnet18,
    #             view_num = 1,
    #             device = 3
    #         )
    #     ],
    #     demos = dict(
    #         task_name = 'bowl_picking',
    #         expert_demo_nums = [],
    #         view_num = 1  
    #     )
    # )

    # bowl_unstacking_encoders = [
    #     load_encoder(**encoder_args) for encoder_args in bowl_unstacking_info['encoders']
    # ]
    # print('LOADED THE ENCODERS')
    # # bowl_unstacking_demos = [load_expert_demos_per_task(**bowl_unstacking_info['demos']) for i in range(4)]
    # bowl_unstacking_demos = load_expert_demos_per_task(**bowl_unstacking_info['demos'])
    # print('LOADED THE DEMOS')


    # for encoder_id, encoder in enumerate(bowl_unstacking_encoders):
    #     score_matrix = calc_encoder_score(
    #         encoder = encoder,
    #         all_expert_demos = bowl_unstacking_demos, 
    #         encoder_id = encoder_id,
    #         device = encoder_id
    #     )
    #     encoder_score = np.sum(score_matrix)

    #     print('id: {} encoder_score: {}'.format(encoder_id, encoder_score))
    #     print('----')
    #     _ = input("Press Enter to continue... after reseting env")
        # import ipdb; ipdb.set_trace()
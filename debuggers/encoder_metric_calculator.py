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

from itertools import combinations, permutations
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

def get_expert_representations_per_encoder(encoder, task_expert_demos, representation_type='image', device=1):
    # Traverse through all the experts and get the representations
    task_representations = []
    for expert_id in range(len(task_expert_demos)):
        expert_representations = []
        task_len = len(task_expert_demos[expert_id]['image_obs'])
        
        for batch_id in range(0, task_len, 10): # in order to prevent cuda out of memory error we load the demos in batches
            batch_reprs = []
            if 'image' in representation_type:
                new_repr = encoder(task_expert_demos[expert_id]['image_obs'][batch_id:min(batch_id+10, task_len),:].to(device)).detach().cpu()
                batch_reprs.append(new_repr)
            if 'tactile' in representation_type:
                batch_reprs.append(
                    task_expert_demos[expert_id]['tactile_repr'][batch_id:min(batch_id+10, task_len),:].detach().cpu()
                )

            if 'state' in representation_type:
                batch_reprs.append(
                    task_expert_demos[expert_id]['robot_state'][batch_id:min(batch_id+10, task_len),:]
                )
            
            batch_repr = torch.concat(batch_reprs, -1) # Concatenate them from the end dimension

            expert_representations.append(batch_repr)
        expert_representations = torch.concat(expert_representations, 0)
        print('expert_representations.shape: {}'.format(expert_representations.shape))
        task_representations.append(expert_representations)
    
    return task_representations

def calc_representation_score(encoder, all_expert_demos, representation_type='image', device=1): # Will get all the representations and calculate the score of the 

    all_expert_representations = get_expert_representations_per_encoder(
        encoder = encoder,
        task_expert_demos = all_expert_demos,
        device = device,
        representation_type = representation_type
    )

    scores_dict = dict()
    traj_idx = range(len(all_expert_representations))
    # traj_idx_comb = combinations(traj_idx, 2)
    traj_idx_comb = permutations(traj_idx, 2)

    num_scores = 0
    total_score = 0
    for i, j in traj_idx_comb:
        traj1 = all_expert_representations[i]
        traj2 = all_expert_representations[j]

        key = f'{i}_{j}'
        traj_score = calc_traj_score(traj1, traj2)
        scores_dict[key] = traj_score
        num_scores += 1
        total_score += traj_score

    print('SCORE DICT: {}'.format(scores_dict))

    return scores_dict, total_score / num_scores

# This image transform will have everything
def load_expert_demos_per_task(task_name, expert_demo_nums, tactile_repr_module, view_num=0):
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
    tactile_reprs = []
    robot_states = []
    old_demo_id = -1
    for step_id in range(len(data['image']['indices'])): 
        demo_id, image_id = data['image']['indices'][step_id]
        if (demo_id != old_demo_id and step_id > 0) or (step_id == len(data['image']['indices'])-1): # NOTE: We are losing the last frame of the last expert

            expert_demos.append(dict(
                image_obs = torch.stack(image_obs, 0), 
                tactile_repr = torch.stack(tactile_reprs, 0),
                robot_state = torch.stack(robot_states, 0)
            ))
            image_obs = [] 

        # Get image representations
        image = load_dataset_image(
            data_path = data_path, 
            demo_id = demo_id, 
            image_id = image_id,
            view_num = view_num,
            transform = image_transform
        )
        image_obs.append(image)

        # Get the tactile representation
        _, tactile_id = data['tactile']['indices'][step_id]
        tactile_value = data['tactile']['values'][demo_id][tactile_id]
        tactile_repr = tactile_repr_module.get(tactile_value, detach=False)
        tactile_reprs.append(tactile_repr)

        # Set actions
        _, allegro_id = data['allegro_joint_states']['indices'][step_id]
        allegro_state = torch.FloatTensor(data['allegro_joint_states']['values'][demo_id][allegro_id])
        
        # Set kinova action 
        _, kinova_id = data['kinova']['indices'][step_id]
        kinova_state = torch.FloatTensor(data['kinova']['values'][demo_id][kinova_id][:-4])
        robot_state = torch.concat([allegro_state, kinova_state], axis=-1)
        robot_states.append(robot_state)

        old_demo_id = demo_id

    return expert_demos


def load_encoder(model_type, model_path, device=1, view_num=0, encoder_fn=None, model_name=None):
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

# @hydra.main(version_base=None, config_path='../tactile_learning/configs', config_name='debug')
# def get_encoder_score(cfg: DictConfig): # This will be used to set the model path and etc
#     cfg = cfg.encoder_metric_calculator
#     task_info = dict(
#         encoders = [dict(
#             model_path = cfg.model_path,
#             model_type = cfg.model_type,
#             view_num = cfg.view_num,
#             encoder_fn = cfg.encoder_fn,
#             device = cfg.device
#         )],
#         demo = dict(
#             task_name = cfg.task_name,
#             expert_demo_nums = cfg.expert_demo_nums,
#             view_num = cfg.view_num
#         )
#     )

#     encoders = [
#         load_encoder(**encoder_args) for encoder_args in task_info['encoders']
#     ]
#     print('LOADED THE ENCODERS')
#     # bowl_unstacking_demos = [load_expert_demos_per_task(**bowl_unstacking_info['demos']) for i in range(4)]
#     demos = load_expert_demos_per_task(**task_info['demos'])
#     print('LOADED THE DEMOS')

#     for encoder_id, encoder in enumerate(encoders):
#         score_matrix = calc_representation_score(
#             encoder = encoder,
#             all_expert_demos = demos, 
#             encoder_id = encoder_id,
#             device = encoder_id
#         )
#         encoder_score = np.sum(score_matrix)

#         print('id: {} encoder_score: {}'.format(encoder_id, encoder_score))
#         print('----')
#         _ = input("Press Enter to continue... ")


def get_tactile_repr_module(tactile_out_dir, device=1):
    tactile_cfg, tactile_encoder, _ = init_encoder_info(
        device = device,
        out_dir = tactile_out_dir,
        encoder_type = 'tactile',
        model_type='byol'
    )
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

    return tactile_repr_module

if __name__ == '__main__':
    # First let's try
    # get_encoder_score()


    plier_picking_info = dict(
        encoders = [
            dict(
                model_name = 'Joint Temporal',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.07/14-18_temporal_ssl_plier_picking_view_0_joint_resnet',
                model_type = 'temporal'
            ),
            dict(
                model_name = 'Contrastive Temporal',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.07/14-19_temporal_ssl_plier_picking_view_0_contrastive_resnet',
                model_type = 'temporal'
            ),
            dict(
                model_name = 'Temporal All',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.02/21-37_temporal_ssl_bs_32_epochs_1000_lr_1e-05_plier_picking_frame_diff_8',
                model_type = 'temporal'
            ),
            dict(
                model_name = 'BYOL Resnet',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.07/18-14_image_byol_plier_picking_view_0_resnet',
                model_type = 'byol'
            ),
            dict(
                model_name = 'BC Resnet',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.07/21-18_bc_plier_picking_view_0_resnet',
                model_type = 'bc'
            ),
            dict(
                model_name = 'Pretrained Resnet',
                model_path = None, 
                encoder_fn = resnet18,
                model_type = 'pretrained'
            )
        ],
        demos = dict(
            task_name = 'plier_picking',
            expert_demo_nums = [3,10,15,16,20,25],
            view_num = 0,
            tactile_repr_module = get_tactile_repr_module(
                tactile_out_dir = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.01.28/12-32_tactile_byol_bs_512_tactile_play_data_alexnet_pretrained_duration_120',
                device = 1
            )
        )
    )

    bowl_picking_info = dict(
        encoders = [
            dict(
                model_name = 'Joint Temporal',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.07/21-02_temporal_ssl_bowl_picking_view_1_resnet',
                model_type = 'temporal',
                view_num = 1
            ),
            dict(
                model_name = 'Contrastive Temporal',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.07/17-26_temporal_ssl_bowl_picking_view_1_contrastive_resnet',
                model_type = 'temporal',
                view_num = 1
            ),
            dict(
                model_name = 'Temporal All',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.06/18-27_temporal_ssl_bs_32_epochs_1000_view_1_bowl_picking_frame_diff_5_resnet',
                model_type = 'temporal',
                view_num = 1
            ),
            dict(
                model_name = 'BYOL Resnet',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.05.06/10-50_image_byol_bs_32_epochs_500_lr_1e-05_bowl_picking_after_rss',
                model_type = 'byol',
                view_num = 1
            ),
            dict(
                model_name = 'BC Resnet',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.07/21-20_bc_bowl_picking_view_1_resnet',
                model_type = 'bc',
                view_num = 1
            ),
            dict(
                model_name = 'Pretrained Resnet',
                model_path = None, 
                encoder_fn = resnet18,
                model_type = 'pretrained',
                view_num = 1
            )
        ],
        demos = dict(
            task_name = 'bowl_picking',
            expert_demo_nums = [],
            view_num = 1,
            tactile_repr_module = get_tactile_repr_module(
                tactile_out_dir = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.01.28/12-32_tactile_byol_bs_512_tactile_play_data_alexnet_pretrained_duration_120',
                device = 1
            )
        )
    )

    card_turning_info = dict(
        encoders = [
            dict(
                model_name = 'Joint Temporal',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.07/15-13_temporal_ssl_card_turning_view_0_joint_resnet',
                model_type = 'temporal',
            ),
            dict(
                model_name = 'Contrastive Temporal',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.07/14-21_temporal_ssl_card_turning_view_0_contrastive_resnet',
                model_type = 'temporal'
            ),
            dict(
                model_name = 'Temporal All',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.05/19-59_temporal_ssl_bs_32_epochs_1000_lr_1e-05_card_turning_frame_diff_5_resnet',
                model_type = 'temporal'
            ),
            dict(
                model_name = 'BYOL Resnet',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.07/17-44_image_byol_card_turning_view_0_resnet',
                model_type = 'byol',
            ),
            dict(
                model_name = 'BC Resnet',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.07/21-37_bc_card_turning_view_0_resnet',
                model_type = 'bc',
            ),
            dict(
                model_name = 'Pretrained Resnet',
                model_path = None, 
                encoder_fn = resnet18,
                model_type = 'pretrained',
            )
        ],
        demos = dict(
            task_name = 'card_turning',
            expert_demo_nums = [],
            view_num = 0,
            tactile_repr_module = get_tactile_repr_module(
                tactile_out_dir = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.01.28/12-32_tactile_byol_bs_512_tactile_play_data_alexnet_pretrained_duration_120',
                device = 1
            )
        )
    )

    card_flipping_info = dict(
        encoders = [
            dict(
                model_name = 'Joint Temporal',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.07/14-41_temporal_ssl_card_flipping_view_0_joint_resnet',
                model_type = 'temporal',
            ),
            dict(
                model_name = 'Contrastive Temporal',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.07/14-41_temporal_ssl_card_flipping_view_0_contrastive_resnet',
                model_type = 'temporal'
            ),
            dict(
                model_name = 'Temporal All',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.05.31/21-20_temporal_ssl_bs_32_epochs_1000_lr_1e-05_card_flipping_frame_diff_8',
                model_type = 'temporal'
            ),
            dict(
                model_name = 'BYOL Resnet',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.07/17-45_image_byol_card_flipping_view_0_resnet',
                model_type = 'byol',
            ),
            dict(
                model_name = 'BC Resnet',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.07/21-45_bc_card_flipping_view_0_resnet',
                model_type = 'bc',
            ),
            dict(
                model_name = 'Pretrained Resnet',
                model_path = None, 
                encoder_fn = resnet18,
                model_type = 'pretrained',
            )
        ],
        demos = dict(
            task_name = 'card_flipping',
            expert_demo_nums = [],
            view_num = 0,
            tactile_repr_module = get_tactile_repr_module(
                tactile_out_dir = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.01.28/12-32_tactile_byol_bs_512_tactile_play_data_alexnet_pretrained_duration_120',
                device = 0
            )
        )
    )

    peg_insertion_info = dict(
        encoders = [
            dict(
                model_name = 'Joint Temporal',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.07/14-40_temporal_ssl_peg_insertion_view_0_joint_resnet',
                model_type = 'temporal',
            ),
            dict(
                model_name = 'Contrastive Temporal',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.07/15-05_temporal_ssl_peg_insertion_view_0_contrastive_resnet',
                model_type = 'temporal'
            ),
            dict(
                model_name = 'Temporal All',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.06/22-59_temporal_ssl_bs_32_epochs_1000_view_0_peg_insertion_frame_diff_5_resnet',
                model_type = 'temporal'
            ),
            dict(
                model_name = 'BYOL Resnet',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.07/18-38_image_byol_peg_insertion_view_0_resnet',
                model_type = 'byol',
            ),
            dict(
                model_name = 'BC Resnet',
                model_path = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.06.07/21-18_bc_peg_insertion_view_0_resnet',
                model_type = 'bc',
            ),
            dict(
                model_name = 'Pretrained Resnet',
                model_path = None, 
                encoder_fn = resnet18,
                model_type = 'pretrained',
            )
        ],
        demos = dict(
            task_name = 'peg_insertion',
            expert_demo_nums = [],
            view_num = 0,
            tactile_repr_module = get_tactile_repr_module(
                tactile_out_dir = '/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.01.28/12-32_tactile_byol_bs_512_tactile_play_data_alexnet_pretrained_duration_120',
                device = 1
            )
        )
    )

    task_infos = {
        'Bowl Unstacking': bowl_picking_info,
        'Plier Picking': plier_picking_info,
        'Eraser Turning': card_turning_info, 
        'Sponge Flipping': card_flipping_info,
        'Peg Insertion': peg_insertion_info
    }

    for task_name, task_info in task_infos.items():
        print(f'LOADING ENCODERS FOR {task_name}, {task_info}')
        # task_encoders = [
        #     load_encoder(**encoder_args) for encoder_args in task_info['encoders']
        # ]
        task_encoders = []
        for encoder_args in task_info['encoders']:
            print(
                f'LOADING: {encoder_args}'
            )
            task_encoders.append(
                load_encoder(**encoder_args))
            
        task_demos = load_expert_demos_per_task(**task_info['demos'])

        encoder = task_encoders[2] # This is the main function

        _, tactile_score = calc_representation_score(
            encoder = encoder,
            all_expert_demos = task_demos,
            device = 1,
            representation_type = 'state' # It could be only tactile
        )

        print(f'TASK: {task_name} - STATE REPR SCORE: {tactile_score}')

        # encoder_scores = []
        # for encoder_id, encoder in enumerate(task_encoders):
        #     score_dict, encoder_score = calc_representation_score(
        #         encoder = encoder,
        #         all_expert_demos = task_demos,
        #         device = 1,
        #         representation_type = 'image_tactile' # It could be only tactile
        #     )

        #     encoder_scores.append(encoder_score)
        #     print(f'TASK NAME: {task_name}, ENCODER NAME: {task_info["encoders"][encoder_id]["model_name"]}, SCORE: {encoder_score}')
        #     print('---')
        #     _ = input('Press Enter to go to the next encoder')

        # print(f'***** TASK {task_name} COMPLETED - SCORES: {encoder_scores}\n-----------------')
        _ = input('Press to enter to the next task')
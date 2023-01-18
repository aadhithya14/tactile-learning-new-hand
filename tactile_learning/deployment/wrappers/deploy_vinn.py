# Helper script to load models
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image as im
from omegaconf import OmegaConf
from tqdm import tqdm 
from torchvision.datasets.folder import default_loader as loader

from holobot.constants import *
from holobot.utils.network import ZMQCameraSubscriber
from holobot.robot.allegro.allegro_kdl import AllegroKDL

from tactile_learning.deployment.load_models import load_model
from tactile_learning.deployment.nn_buffer import NearestNeighborBuffer
from tactile_learning.models.knn import KNearestNeighbors, ScaledKNearestNeighbors
from tactile_learning.utils.augmentations import crop_transform
from tactile_learning.utils.constants import *
from tactile_learning.utils.data import load_data
from tactile_learning.utils.tactile_image import get_tactile_image
from tactile_learning.utils.visualization import *


class DeployVINN:
    def __init__(
        self,
        tactile_out_dir,
        image_out_dir,
        data_path,
        deployment_run_name,
        task_object,
        robots = ['allegro', 'kinova'],
        representation_types = ['image', 'tactile', 'kinova', 'allegro'],
        use_encoder = True,
        nn_buffer_size=100,
        nn_k=20,
        demos_to_use=[0],
        run_the_demo=False,
        sensor_indices = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14),
        allegro_finger_indices = (0,1,2,3),
        single_sensor_tactile=False,
        stacked_tactile=False,
        alexnet_tactile=False
    ):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29504"

        torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
        torch.cuda.set_device(0)

        self.sensor_indices = sensor_indices 
        self.allegro_finger_indices = [j for i in allegro_finger_indices for j in range(i*3,(i+1)*3)]
        self.use_encoder = use_encoder
        self.representation_types = representation_types
        print('self.representation_types: {}, len(demos_to_use): {}'.format(
            self.representation_types, len(demos_to_use)
        ))
        self._set_thumb_value(task_object)
        if run_the_demo:
            assert len(demos_to_use) == 1, 'While running one demo, length of the demos to use should be 1'
        self.run_the_demo = run_the_demo # Boolean to indicate if 
        self.demos_to_use = demos_to_use
        self.robots = robots # Types of robots to be used in the states

        device = torch.device('cuda:0')
        self.single_sensor_tactile = single_sensor_tactile
        self.stacked_tactile = stacked_tactile
        self.alexnet_tactile = alexnet_tactile
        self.tactile_cfg, self.tactile_encoder, self.tactile_transform = self._init_encoder_info(device, tactile_out_dir, 'tactile')
        self.tactile_repr_size = self.tactile_cfg.encoder.out_dim * len(sensor_indices) if single_sensor_tactile else self.tactile_cfg.encoder.out_dim
        self.image_cfg, self.image_encoder, self.image_transform = self._init_encoder_info(device, image_out_dir, 'image')
        self.inv_image_transform = self._get_inverse_image_norm()

        roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
        self.data_path = data_path
        self.data = load_data(roots, demos_to_use=demos_to_use) # This will return all the desired indices and the values
        self._get_all_representations()
        self.state_id = 0 # Increase it with each get_action

        self.kdl_solver = AllegroKDL()
        self.buffer = NearestNeighborBuffer(nn_buffer_size)
        self.nn_k = nn_k
        self.knn = ScaledKNearestNeighbors(
            self.all_representations, # Both the input and the output of the nearest neighbors are
            self.all_representations,
            representation_types,
            self.tactile_repr_size
        )

        self.deployment_dump_dir = os.path.join('/home/irmak/Workspace/Holo-Bot/deployment_data', deployment_run_name)
        os.makedirs(self.deployment_dump_dir, exist_ok=True)
        self.deployment_info = dict(
            all_representations = self.all_representations,
            curr_representations = [], # representations will be appended to this list
            closest_representations = []
        )

    def _set_thumb_value(self, task_object):
        if task_object == 'box_handle_lifting':
            self.set_thumb_values = ALLEGRO_BOX_HANDLE_LIFTING_THUMB_VALUES
        else:
            self.set_thumb_values = None

    def _init_encoder_info(self, device, out_dir, encoder_type='tactile'): # encoder_type: either image or tactile
        if encoder_type == 'tactile' and self.alexnet_tactile:
            # Load alexnet and create a mock cfg to get the out_dim
            encoder = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
            encoder.fc = nn.Identity()
            cfg = OmegaConf.create({"encoder":{"out_dim":1000}})
        else:
            cfg = OmegaConf.load(os.path.join(out_dir, '.hydra/config.yaml'))
            model_path = os.path.join(out_dir, 'models/byol_encoder_best.pt')
            encoder = load_model(cfg, device, model_path)
        encoder.eval() 
        if encoder_type == 'tactile':
            if self.stacked_tactile:
                transform = T.Compose([
                    T.Resize((cfg.tactile_image_size, cfg.tactile_image_size)),
                    T.Normalize(TACTILE_IMAGE_MEANS*15, TACTILE_IMAGE_STDS*15),
                ])
            elif self.alexnet_tactile:
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]) # These are the requirements for alexnet
            else:
                transform = T.Compose([
                    T.Resize((cfg.tactile_image_size, cfg.tactile_image_size)),
                    T.Normalize(TACTILE_IMAGE_MEANS, TACTILE_IMAGE_STDS),
                ])
        elif encoder_type == 'image':
            transform = T.Compose([
                T.Resize((480,640)),
                T.Lambda(crop_transform),
                T.ToTensor(),
                T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS),
            ]) 

        return cfg, encoder, transform

    def _get_inverse_image_norm(self):
        np_means = np.asarray(VISION_IMAGE_MEANS)
        np_stds = np.asarray(VISION_IMAGE_STDS)

        inv_normalization_transform = T.Compose([
            T.Normalize(mean = [0,0,0], std = 1 / np_stds ), 
            T.Normalize(mean = -np_means, std = [1,1,1])
        ])

        return inv_normalization_transform

    def _get_tactile_image(self, tactile_value):
        tactile_image = get_tactile_image(tactile_value)
        return self.tactile_transform(tactile_image)

    def _get_curr_image(self, host='172.24.71.240', port=10005):
        image_subscriber = ZMQCameraSubscriber(
            host = host,
            port = port,
            topic_type = 'RGB'
        )
        image, _ = image_subscriber.recv_rgb_image()
        print('get_curr_image.shape: {}'.format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = im.fromarray(image)
        img = self.image_transform(image)
        return torch.FloatTensor(img)

    def _load_dataset_image(self, demo_id, image_id):
        roots = glob.glob(f'{self.data_path}/demonstration_*')
        roots = sorted(roots)
        image_root = roots[demo_id]
        image_path = os.path.join(image_root, 'cam_0_rgb_images/frame_{}.png'.format(str(image_id).zfill(5)))
        img = self.image_transform(loader(image_path))
        return torch.FloatTensor(img)

    def _get_tactile_representation_with_stacked_tactile_encoder(self, tactile_values):
        def _get_stacked_tactile_image(tactile_values):
            tactile_image = torch.FloatTensor(tactile_values)
            tactile_image = tactile_image.view(15,4,4,3) # Just making sure that everything stays the same
            tactile_image = torch.permute(tactile_image, (0,3,1,2))
            tactile_image = tactile_image.reshape(-1,4,4)
            return self.tactile_transform(tactile_image)

        tactile_image = _get_stacked_tactile_image(tactile_values)
        return self.tactile_encoder(tactile_image.unsqueeze(0)).squeeze()

    def _get_tactile_representation_with_single_sensor_encoder(self, tactile_values): # tactile_values.shape: (15,16,3)
        def _get_single_tactile_image(tactile_value):
            tactile_image = torch.FloatTensor(tactile_value) # tactile_value.shape: (16,3)
            tactile_image = tactile_image.view(4,4,3)
            tactile_image = torch.permute(tactile_image, (2,0,1))
            return self.tactile_transform(tactile_image)

        for sensor_id in range(len(tactile_values)):
            curr_tactile_value = tactile_values[sensor_id]
            curr_tactile_image = _get_single_tactile_image(curr_tactile_value).unsqueeze(0) # To make it as if it's a batch
            if sensor_id == 0:
                curr_repr = self.tactile_encoder(curr_tactile_image).squeeze() # shape: (64)
            else:
                curr_repr =  torch.cat([curr_repr, self.tactile_encoder(curr_tactile_image).squeeze()], dim=0)

        return curr_repr

    def _get_alexnet_tactile_representation(self, tactile_values): # This is dependent on
        def _get_whole_hand_tactile_image(tactile_values): 
            # tactile_values: (15,16,3) - turn it into 16,16,3 by concatenating 0z
            tactile_image = torch.FloatTensor(tactile_values)
            tactile_image = F.pad(tactile_image, (0,0,0,0,1,0), 'constant', 0)
            # reshape it to 4x4
            tactile_image = tactile_image.view(16,4,4,3)

            # concat for it have its proper shape
            tactile_image = torch.concat([
                torch.concat([tactile_image[i*4+j] for j in range(4)], dim=0)
                for i in range(4)
            ], dim=1)

            tactile_image = torch.permute(tactile_image, (2,0,1))
            pre_tactile_transform = T.Compose([
                T.Resize((16,16)),
                T.Normalize(TACTILE_IMAGE_MEANS, TACTILE_IMAGE_STDS)
            ])
            return pre_tactile_transform(tactile_image)
        
        tactile_image = _get_whole_hand_tactile_image(tactile_values)
        # map it bw 0 and 1 (that's needed for alexnet)
        min_tactile = -112.1339
        max_tactile = 99.1087 # These are calculated - it could effect how things are NOTE: should be careful here! 
        tactile_image = (tactile_image - min_tactile) / (max_tactile - min_tactile)
        # then transform as required for alexnet
        tactile_image = self.tactile_transform(tactile_image)
        return self.tactile_encoder(tactile_image.unsqueeze(0)).squeeze()

    # tactile_values: (N,16,3) - N: number of sensors
    # robot_states: { allegro: allegro_tip_positions: (3*M,) - 3 values for each finger M: number of fingers included,
    #                 kinova: kinova_states : (3,) - cartesian position of the arm end effector}
    def _get_one_representation(self, image, tactile_values, robot_states):
        for i,repr_type in enumerate(self.representation_types):
            if repr_type == 'allegro' or repr_type == 'kinova':
                new_repr = robot_states[repr_type]
            elif repr_type == 'tactile':
                if self.use_encoder:
                    if self.single_sensor_tactile:
                        new_repr = self._get_tactile_representation_with_single_sensor_encoder(tactile_values).detach().cpu().numpy()
                    elif self.stacked_tactile: # - TODO
                        new_repr = self._get_tactile_representation_with_stacked_tactile_encoder(tactile_values).detach().cpu().numpy()
                    elif self.alexnet_tactile:
                        new_repr = self._get_alexnet_tactile_representation(tactile_values).detach().cpu().numpy()
                    else:
                        tactile_image = self._get_tactile_image(tactile_values).unsqueeze(dim=0)
                        new_repr = self.tactile_encoder(tactile_image)
                        new_repr = new_repr.detach().cpu().numpy().squeeze()
                else:
                    new_repr = tactile_values.flatten()
            elif repr_type == 'image':
                new_repr = self.image_encoder(image.unsqueeze(dim=0)) # Add a dimension to the first axis so that it could be considered as a batch
                new_repr = new_repr.detach().cpu().numpy().squeeze()

            if i == 0:
                curr_repr = new_repr 
            else: 
                curr_repr = np.concatenate([curr_repr, new_repr], axis=0)
                
            # print('in _get_one_representation - curr_repr.shape: {}'.format(curr_repr.shape))
        
        return curr_repr

    def _get_all_representations(self):  
        print('Getting all representations')
        # For each tactile value and allegro tip position 
        # get one representation and add it to all representations
        repr_dim = 0
        if 'tactile' in self.representation_types:
            if self.use_encoder:
                repr_dim = self.tactile_repr_size
            else:
                repr_dim = len(self.sensor_indices) * 16 * 3
        if 'allegro' in self.representation_types:  repr_dim += len(self.allegro_finger_indices)
        if 'kinova' in self.representation_types: repr_dim += 7
        if 'image' in self.representation_types: repr_dim += 512 # NOTE: This should be lower (or tactile should be higher) - we could use a random layer on top?

        self.all_representations = np.zeros((
            len(self.data['tactile']['indices']), repr_dim
        ))

        print('all_representations.shape: {}'.format(self.all_representations.shape))
        pbar = tqdm(total=len(self.data['tactile']['indices']))

        for index in range(len(self.data['tactile']['indices'])):
            demo_id, tactile_id = self.data['tactile']['indices'][index]
            _, allegro_tip_id = self.data['allegro_tip_states']['indices'][index]
            _, kinova_id = self.data['kinova']['indices'][index]
            _, image_id = self.data['image']['indices'][index]

            tactile_value = self.data['tactile']['values'][demo_id][tactile_id][self.sensor_indices,:,:] # This should be (N,16,3)
            allegro_tip_position = self.data['allegro_tip_states']['values'][demo_id][allegro_tip_id] # This should be (M*3,)
            kinova_state = self.data['kinova']['values'][demo_id][kinova_id]
            image = self._load_dataset_image(demo_id, image_id)
            
            robot_states = dict(
                allegro = allegro_tip_position,
                kinova = kinova_state
            )
            representation = self._get_one_representation(
                image,
                tactile_value, 
                robot_states
            )
            # Add only tip positions as the representation
            self.all_representations[index, :] = representation[:]
            pbar.update(1)

        pbar.close()

    # Method that will save all representations and each representation in each timestamp
    def save_deployment(self):
        print('saving deployment - deployment_info[all_repr].shape: {}, deployment_info[curr_reprs].shape: {}'.format(
            self.deployment_info['all_representations'].shape, len(self.deployment_info['curr_representations'])
        ))
        with open(os.path.join(self.deployment_dump_dir, 'deployment_info.pkl'), 'wb') as f:
            pickle.dump(self.deployment_info, f)


    def get_action(self, tactile_values, recv_robot_state, visualize=False):
        if self.run_the_demo:
            action = self._get_demo_action()
        else:
            action = self._get_knn_action(tactile_values, recv_robot_state, visualize)
        return action

    def _get_demo_action(self):
        demo_id, action_id = self.data['allegro_actions']['indices'][self.state_id] 
        allegro_action = self.data['allegro_actions']['values'][demo_id][action_id] # Get the next commanded action (commanded actions are saved in that timestamp)
        if self.set_thumb_values is not None:
            allegro_action[-4:] = self.set_thumb_values
        action = dict(
            allegro = allegro_action
        )
        
        if 'kinova' in self.robots:
            _, kinova_id = self.data['kinova']['indices'][self.state_id] 
            kinova_action = self.data['kinova']['values'][demo_id][kinova_id] # Get the next saved kinova_state
            action['kinova'] = kinova_action

        print(f'applying the {self.state_id}th demo action: {action}')

        self.state_id += 1

        return action

    # tactile_values.shape: (16,15,3)
    # robot_state: {allegro: allegro_joint_state (16,), kinova: kinova_cart_state (3,)}
    def _get_knn_action(self, tactile_values, recv_robot_state, visualize=False):
        # Get the allegro tip positions with kdl solver 
        allegro_joint_state = recv_robot_state['allegro']
        fingertip_positions = self.kdl_solver.get_fingertip_coords(allegro_joint_state) # - fingertip position.shape: (12)
        curr_fingertip_position = fingertip_positions[self.allegro_finger_indices]
        kinova_cart_state = recv_robot_state['kinova']
        curr_robot_state = dict(
            allegro = curr_fingertip_position,
            kinova = kinova_cart_state
        )

        # Get the tactile image from the tactile values
        curr_tactile_values = tactile_values[self.sensor_indices,:,:]

        # Get the current visual image
        image = self._get_curr_image()

        # Get the representation with the given tactile value
        curr_representation = self._get_one_representation(
            image,
            curr_tactile_values, 
            curr_robot_state
        )

        self.deployment_info['curr_representations'].append(curr_representation)
        _, nn_idxs, nn_separate_dists = self.knn.get_k_nearest_neighbors(curr_representation, k=self.nn_k)
        closest_representation = self.all_representations[nn_idxs[0]]
        self.deployment_info['closest_representations'].append(closest_representation)

        # Choose the action with the buffer 
        id_of_nn = self.buffer.choose(nn_idxs)
        nn_id = nn_idxs[id_of_nn]
        if nn_id+1 >= len(self.data['allegro_actions']['indices']):
            nn_idxs = np.delete(nn_idxs, id_of_nn)
            id_of_nn = self.buffer.choose(nn_idxs)
            nn_id = nn_idxs[id_of_nn]
        demo_id, action_id = self.data['allegro_actions']['indices'][nn_id+1] 
        nn_allegro_action = self.data['allegro_actions']['values'][demo_id][action_id] # Get the next commanded action (commanded actions are saved in that timestamp)
        if self.set_thumb_values is not None:
            nn_allegro_action[-4:] = self.set_thumb_values
        nn_action = dict(
            allegro = nn_allegro_action
        )
        
        if 'kinova' in self.robots:
            _, kinova_id = self.data['kinova']['indices'][nn_id+1] # We send it to go to the next state
            nn_kinova_action = self.data['kinova']['values'][demo_id][kinova_id] # Get the next saved kinova_state
            nn_action['kinova'] = nn_kinova_action
        print('STATE ID: {}, CHOSEN DEMO ID: {}, ALLEGRO ID: {}, KINOVA ID: {}, SEPARATE_DISTS[ID_OF_NN:{}]: {}'.format(
            self.state_id, demo_id, action_id, kinova_id, id_of_nn, nn_separate_dists[id_of_nn]
        ))

        # Visualize if given 
        if visualize: # TODO: This should be fixed for the whole robot
            self._visualize_state(
                tactile_values, # We do want to plot all the tactile values - not only the ones we want  
                curr_fingertip_position,
                kinova_cart_state[:3],
                id_of_nn,
                nn_idxs,
                nn_separate_dists, # We'll visualize 3 more neighbors' distances with their demos and ids

            )

        self.state_id += 1

        return nn_action

    def _visualize_state(self, curr_tactile_values, curr_fingertip_position, curr_kinova_cart_pos, id_of_nn, nn_idxs, nn_separate_dists):
        # Get the current image 
        curr_image = self.inv_image_transform(self._get_curr_image()).numpy().transpose(1,2,0)
        curr_image_cv2 = cv2.cvtColor(curr_image*255, cv2.COLOR_RGB2BGR)
        print('curr_image.shape: {}'.format(curr_image.shape))

        nn_id = nn_idxs[id_of_nn]
        # Get the next visualization data
        knn_vis_data = self._get_data_with_id_for_visualization(nn_id)
        prev_knn_vis_data = self._get_data_with_id_for_visualization(nn_id-1)
        next_knn_vis_data = self._get_data_with_id_for_visualization(nn_id+1)

        # Get the demo ids of the closest 3 neighbor
        demo_ids = []
        viz_id_of_nns = []
        for i in range(3):
            if not (id_of_nn+i >= len(nn_idxs)):
                viz_nn_id = nn_idxs[id_of_nn+i]
                viz_id_of_nns.append(id_of_nn+i)
            else:
                viz_id_of_nns.append(viz_id_of_nns[-1])
            demo_id, _ = self.data['tactile']['indices'][viz_nn_id]
            demo_ids.append(demo_id)

        if not ('image' in self.representation_types):
            # Dump all the current state, nn state and curr image
            dump_camera_image()
            dump_whole_state(curr_tactile_values, curr_fingertip_position, curr_kinova_cart_pos, title='curr_state')
            dump_whole_state(knn_vis_data['tactile'], knn_vis_data['allegro'], knn_vis_data['kinova'], title='knn_state')
            # Plot from the dumped images
            dump_knn_state(
                dump_dir = self.deployment_dump_dir,
                img_name = 'state_{}.png'.format(str(self.state_id).zfill(2))
            )
    
        else:
            dump_whole_state(curr_tactile_values, curr_fingertip_position, curr_kinova_cart_pos, title='curr_state', vision_state=curr_image_cv2)
            dump_whole_state(knn_vis_data['tactile'], knn_vis_data['allegro'], knn_vis_data['kinova'], title='knn_state', vision_state=knn_vis_data['image'])
            dump_whole_state(prev_knn_vis_data['tactile'], prev_knn_vis_data['allegro'], prev_knn_vis_data['kinova'], title='prev_knn_state', vision_state=prev_knn_vis_data['image'])
            dump_whole_state(next_knn_vis_data['tactile'], next_knn_vis_data['allegro'], next_knn_vis_data['kinova'], title='next_knn_state', vision_state=next_knn_vis_data['image'])
            dump_repr_effects(nn_separate_dists, viz_id_of_nns, demo_ids, self.representation_types)
            dump_knn_state(
                dump_dir = self.deployment_dump_dir,
                img_name = 'state_{}.png'.format(str(self.state_id).zfill(2)),
                image_repr = True,
                add_repr_effects = True,
                include_temporal_states = True
            )
            
    def _get_data_with_id_for_visualization(self, id):
        demo_id, tactile_id = self.data['tactile']['indices'][id]
        _, allegro_tip_id = self.data['allegro_tip_states']['indices'][id]
        _, kinova_id = self.data['kinova']['indices'][id]
        _, image_id = self.data['image']['indices'][id]
        tactile_values = self.data['tactile']['values'][demo_id][tactile_id]
        allegro_finger_tip_pos = self.data['allegro_tip_states']['values'][demo_id][allegro_tip_id]
        kinova_cart_pos = self.data['kinova']['values'][demo_id][kinova_id][:3]
        image = self.inv_image_transform(self._load_dataset_image(demo_id, image_id)).numpy().transpose(1,2,0)
        image_cv2 = cv2.cvtColor(image*255, cv2.COLOR_RGB2BGR)

        visualization_data = dict(
            image = image_cv2,
            kinova = kinova_cart_pos, 
            allegro = allegro_finger_tip_pos, 
            tactile = tactile_values
        )

        return visualization_data
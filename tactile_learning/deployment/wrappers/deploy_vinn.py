# Helper script to load models
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision.transforms as T

from PIL import Image as im
from omegaconf import OmegaConf
from tqdm import tqdm 
from torchvision.datasets.folder import default_loader as loader


from holobot.constants import *
from holobot.utils.network import ZMQCameraSubscriber
from holobot.robot.allegro.allegro_kdl import AllegroKDL

from tactile_learning.deployment.load_models import load_model
from tactile_learning.deployment.nn_buffer import NearestNeighborBuffer
from tactile_learning.models.knn import KNearestNeighbors
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
        deployment_dump_dir,
        robots = ['allegro', 'kinova'],
        representation_types = ['image', 'tactile', 'kinova', 'allegro'],
        use_encoder = True,
        nn_buffer_size=100,
        nn_k=20,
        set_thumb_values=None,
        demos_to_use=[0],
        run_the_demo=False,
        sensor_indices = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14),
        allegro_finger_indices = (0,1,2,3),
        run_num=1,
    ):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"

        torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
        torch.cuda.set_device(0)

        self.sensor_indices = sensor_indices 
        self.allegro_finger_indices = [j for i in allegro_finger_indices for j in range(i*3,(i+1)*3)]
        self.use_encoder = use_encoder
        self.representation_types = representation_types
        print('self.representation_types: {}, len(demos_to_use): {}'.format(
            self.representation_types, len(demos_to_use)
        ))
        self.set_thumb_values = set_thumb_values
        if run_the_demo:
            assert len(demos_to_use) == 1, 'While running one demo, length of the demos to use should be 1'
        self.run_the_demo = run_the_demo # Boolean to indicate if 
        self.demos_to_use = demos_to_use
        self.robots = robots # Types of robots to be used in the states

        device = torch.device('cuda:0')
        self.tactile_cfg, self.tactile_encoder, self.tactile_transform = self._init_encoder_info(device, tactile_out_dir, 'tactile')
        self.image_cfg, self.image_encoder, self.image_transform = self._init_encoder_info(device, image_out_dir, 'image')
        self.inv_image_transform = self._get_inverse_image_norm()

        roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
        print('roots: {}'.format(roots))
        self.data_path = data_path
        self.data = load_data(roots, demos_to_use=demos_to_use) # This will return all the desired indices and the values
        self._get_all_representations()
        self.state_id = 0 # Increase it with each get_action

        self.kdl_solver = AllegroKDL()
        self.buffer = NearestNeighborBuffer(nn_buffer_size)
        self.nn_k = nn_k
        self.knn = KNearestNeighbors(
            self.all_representations, # Both the input and the output of the nearest neighbors are
            self.all_representations
        )
        self.run_num = run_num
        # self.deployment_dump_dir = os.path.join(, f'runs/run_{self.run_num}_ue_{self.use_encoder}')
        self.deployment_dump_dir = deployment_dump_dir
        os.makedirs(self.deployment_dump_dir, exist_ok=True)

    def _init_encoder_info(self, device, out_dir, encoder_type='tactile'): # encoder_type: either image or tactile
        cfg = OmegaConf.load(os.path.join(out_dir, '.hydra/config.yaml'))
        if encoder_type == 'tactile':
            cfg.learner_type = 'tactile_byol' # NOTE: This will not be needed in newer trainings - but still
        model_path = os.path.join(out_dir, 'models/byol_encoder.pt')
        encoder = load_model(cfg, device, model_path)
        encoder.eval() 
        if encoder_type == 'tactile':
            transform = T.Compose([
                T.Resize((cfg.tactile_image_size, cfg.tactile_image_size)),
                T.Normalize(TACTILE_IMAGE_MEANS, TACTILE_IMAGE_STDS),
                # T.ToTensor(),
            ])
            # transform = T.Resize(cfg.tactile_image_size, cfg.tactile_image_size)
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

    # tactile_values: (N,16,3) - N: number of sensors
    # robot_states: { allegro: allegro_tip_positions: (3*M,) - 3 values for each finger M: number of fingers included,
    #                 kinova: kinova_states : (3,) - cartesian position of the arm end effector}
    def _get_one_representation(self, image, tactile_values, robot_states):
        for i,repr_type in enumerate(self.representation_types):
            if repr_type == 'allegro' or repr_type == 'kinova':
                new_repr = robot_states[repr_type]
            elif repr_type == 'tactile':
                if self.use_encoder: 
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
                
        return curr_repr

    def _get_all_representations(self):  
        print('Getting all representations')
        # For each tactile value and allegro tip position 
        # get one representation and add it to all representations
        repr_dim = 0
        if 'tactile' in self.representation_types:
            if self.use_encoder:
                repr_dim = self.tactile_cfg.encoder.out_dim
            else:
                repr_dim = len(self.sensor_indices) * 16 * 3
        if 'allegro' in self.representation_types:  repr_dim += len(self.allegro_finger_indices)
        if 'kinova' in self.representation_types: repr_dim += 7
        if 'image' in self.representation_types: repr_dim += 1000 # NOTE: This should be lower (or tactile should be higher) - we could use a random layer on top?

        self.all_representations = np.zeros((
            len(self.data['tactile']['indices']), repr_dim
        ))

        print('all_representations.shape: {}'.format(self.all_representations.shape))
        pbar = tqdm(total=len(self.data['tactile']['indices']))

        for index in range(len(self.data['tactile']['indices'])):
            demo_id, tactile_id = self.data['tactile']['indices'][index]
            _, allegro_tip_id = self.data['allegro_states']['indices'][index]
            _, kinova_id = self.data['kinova']['indices'][index]
            _, image_id = self.data['image']['indices'][index]

            tactile_value = self.data['tactile']['values'][demo_id][tactile_id] # This should be (N,16,3)
            allegro_tip_position = self.data['allegro_states']['values'][demo_id][allegro_tip_id] # This should be (M*3,)
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
        _, nn_idxs = self.knn.get_k_nearest_neighbors(curr_representation, k=self.nn_k)

        # Choose the action with the buffer 
        id_of_nn = self.buffer.choose(nn_idxs)
        nn_id = nn_idxs[id_of_nn]
        demo_id, action_id = self.data['allegro_actions']['indices'][nn_id] 
        nn_allegro_action = self.data['allegro_actions']['values'][demo_id][action_id] # Get the next commanded action (commanded actions are saved in that timestamp)
        if self.set_thumb_values is not None:
            nn_allegro_action[-4:] = self.set_thumb_values
        nn_action = dict(
            allegro = nn_allegro_action
        )
        
        if 'kinova' in self.robots:
            _, kinova_id = self.data['kinova']['indices'][nn_id] 
            nn_kinova_action = self.data['kinova']['values'][demo_id][kinova_id] # Get the next saved kinova_state
            nn_action['kinova'] = nn_kinova_action
        print('STATE ID: {}, CHOSEN DEMO ID: {}, ALLEGRO ID: {}, KINOVA ID: {}'.format(
            self.state_id, demo_id, action_id, kinova_id
        ))

        # Visualize if given 
        if visualize: # TODO: This should be fixed for the whole robot
            self._visualize_state(
                curr_tactile_values, 
                curr_fingertip_position,
                kinova_cart_state[:3],
                nn_id
            )

        self.state_id += 1

        return nn_action

    def _visualize_state(self, curr_tactile_values, curr_fingertip_position, curr_kinova_cart_pos, nn_id):

        demo_id, tactile_id = self.data['tactile']['indices'][nn_id]
        _, allegro_tip_id = self.data['allegro_states']['indices'][nn_id]
        _, kinova_id = self.data['kinova']['indices'][nn_id]
        _, image_id = self.data['image']['indices'][nn_id]
        curr_image = self.inv_image_transform(self._get_curr_image()).numpy().transpose(1,2,0)
        curr_image_cv2 = cv2.cvtColor(curr_image*255, cv2.COLOR_RGB2BGR)
        print('curr_image.shape: {}'.format(curr_image.shape))

        knn_tactile_values = self.data['tactile']['values'][demo_id][tactile_id]
        knn_tip_pos = self.data['allegro_states']['values'][demo_id][allegro_tip_id]
        knn_cart_pos = self.data['kinova']['values'][demo_id][kinova_id][:3]
        knn_image = self.inv_image_transform(self._load_dataset_image(demo_id, image_id)).numpy().transpose(1,2,0)
        knn_image_cv2 = cv2.cvtColor(knn_image*255, cv2.COLOR_RGB2BGR)

        if not ('image' in self.representation_types):
            # Dump all the current state, nn state and curr image
            dump_camera_image()
            dump_whole_state(curr_tactile_values, curr_fingertip_position, curr_kinova_cart_pos, title='curr_state')
            dump_whole_state(knn_tactile_values, knn_tip_pos, knn_cart_pos, title='knn_state')
            # Plot from the dumped images
            dump_knn_state(
                dump_dir = self.deployment_dump_dir,
                img_name = 'state_{}.png'.format(str(self.state_id).zfill(2))
            )
    
        else:
            dump_whole_state(curr_tactile_values, curr_fingertip_position, curr_kinova_cart_pos, title='curr_state', vision_state=curr_image_cv2)
            dump_whole_state(knn_tactile_values, knn_tip_pos, knn_cart_pos, title='knn_state', vision_state=knn_image_cv2)
            dump_knn_state(
                dump_dir = self.deployment_dump_dir,
                img_name = 'state_{}.png'.format(str(self.state_id).zfill(2)),
                image_repr=True
            )
            
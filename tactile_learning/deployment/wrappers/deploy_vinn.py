# Helper script to load models
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle 
import torch
import torchvision.transforms as T

from omegaconf import OmegaConf
from tqdm import tqdm 

from holobot.robot.allegro.allegro_kdl import AllegroKDL
from tactile_learning.deployment.load_models import load_model
from tactile_learning.deployment.nn_buffer import NearestNeighborBuffer
from tactile_learning.models.knn import KNearestNeighbors
from tactile_learning.utils.visualization import dump_camera_image, dump_tactile_state, dump_knn_state
from tactile_learning.utils.tactile_image import get_tactile_image

import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle 
import torch
import torchvision.transforms as T

from omegaconf import OmegaConf
from tqdm import tqdm 

from holobot.robot.allegro.allegro_kdl import AllegroKDL
from tactile_learning.deployment.load_models import load_model
from tactile_learning.deployment.nn_buffer import NearestNeighborBuffer
from tactile_learning.models.knn import KNearestNeighbors
from tactile_learning.utils.visualization import dump_camera_image, dump_tactile_state, dump_knn_state
from tactile_learning.utils.tactile_image import get_tactile_image

import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle 
import torch
import torchvision.transforms as T

from omegaconf import OmegaConf
from tqdm import tqdm 

from holobot.robot.allegro.allegro_kdl import AllegroKDL
from holobot.constants import *
from tactile_learning.deployment.load_models import load_model
from tactile_learning.deployment.nn_buffer import NearestNeighborBuffer
from tactile_learning.models.knn import KNearestNeighbors
from tactile_learning.utils.visualization import dump_camera_image, dump_whole_state, dump_knn_state
from tactile_learning.utils.tactile_image import get_tactile_image

class DeployVINN:
    def __init__(
        self,
        out_dir,
        robots = ['allegro', 'kinova'],
        sensor_indices = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14),
        allegro_finger_indices = (0,1,2,3),
        representation_types = ['tactile', 'kinova', 'allegro'],
        use_encoder = True,
        nn_buffer_size=100,
        nn_k=20,
        set_thumb_values=None,
        demos_to_use=[0],
        run_the_demo=False,
        run_num=1,
    ):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"

        torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
        torch.cuda.set_device(0)

        self.sensor_indices = sensor_indices 
        self.allegro_finger_indices = [j for i in allegro_finger_indices for j in range(i*3,(i+1)*3)]
        print('self.allegro_finger_indices: {}'.format(self.allegro_finger_indices))
        self.use_encoder = use_encoder
        self.representation_types = representation_types
        self.set_thumb_values = set_thumb_values
        if run_the_demo:
            assert len(demos_to_use) == 1, 'While running one demo, length of the demos to use should be 1'
        self.run_the_demo = run_the_demo # Boolean to indicate if 
        self.demos_to_use = demos_to_use
        self.robots = robots # Types of robots to be used in the states
        print('self.repr_types: {}'.format(self.representation_types))

        self.device = torch.device('cuda:0')
        self.out_dir = out_dir
        self.cfg = OmegaConf.load(os.path.join(out_dir, '.hydra/config.yaml'))
        self.data_path = self.cfg.data_dir
        model_path = os.path.join(out_dir, 'models/byol_encoder.pt')

        self.encoder = load_model(self.cfg, self.device, model_path)
        self.encoder.eval() 

        self.resize_transform = T.Resize((self.cfg.tactile_image_size, self.cfg.tactile_image_size))

        self._load_data(demos_to_use)
        self._get_all_representations()
        self.state_id = 0 # Increase it with each get_action

        self.kdl_solver = AllegroKDL()
        self.last_action = None
        self.buffer = NearestNeighborBuffer(nn_buffer_size)
        self.nn_k = nn_k
        self.knn = KNearestNeighbors(
            self.all_representations, # Both the input and the output of the nearest neighbors are
            self.all_representations
        )
        self.run_num = run_num

    def _load_data(self, demos_to_use):
        roots = glob.glob(f'{self.data_path}/demonstration_*')
        roots = sorted(roots)

        self.tactile_indices = [] 
        self.allegro_indices = [] 
        self.kinova_indices = []
        self.allegro_action_indices = [] 

        self.allegro_actions = {}
        self.tactile_values = {}
        self.allegro_tip_positions = {}
        self.kinova_states = {}

        for demo_id, root in enumerate(roots):
            # Load the indices
            demo_num = int(root.split('/')[-1].split('_')[-1])
            if demo_num in demos_to_use:
                print(f'USING DEMO: {demo_num} [DEMO_ID: {demo_id} - IN ROOT: {root}')
                with open(os.path.join(root, 'tactile_indices.pkl'), 'rb') as f:
                    self.tactile_indices += pickle.load(f)
                with open(os.path.join(root, 'allegro_indices.pkl'), 'rb') as f:
                    self.allegro_indices += pickle.load(f)
                with open(os.path.join(root, 'allegro_action_indices.pkl'), 'rb') as f:
                    self.allegro_action_indices += pickle.load(f)
                with open(os.path.join(root, 'kinova_indices.pkl'), 'rb') as f:
                    self.kinova_indices += pickle.load(f)

                # Load the data
                with h5py.File(os.path.join(root, 'allegro_fingertip_states.h5'), 'r') as f:
                    self.allegro_tip_positions[demo_id] = f['positions'][()][:, self.allegro_finger_indices]
                with h5py.File(os.path.join(root, 'allegro_commanded_joint_states.h5'), 'r') as f:
                    self.allegro_actions[demo_id] = f['positions'][()] # Positions are to be learned - since this is a position control
                with h5py.File(os.path.join(root, 'touch_sensor_values.h5'), 'r') as f:
                    self.tactile_values[demo_id] = f['sensor_values'][()][:,self.sensor_indices,:,:]
                with h5py.File(os.path.join(root, 'kinova_cartesian_states.h5'), 'r') as f:
                    state = np.concatenate([f['positions'][()], f['orientations'][()]], axis=1)
                    self.kinova_states[demo_id] = state
# 
        # print('DEMO_ID KEYS: {}'.format(self.kinova_states.keys()))

    def _get_tactile_image(self, tactile_value):
        tactile_image = get_tactile_image(tactile_value)
        return self.resize_transform(tactile_image)

    # tactile_values: (N,16,3) - N: number of sensors
    # robot_states: { allegro: allegro_tip_positions: (3*M,) - 3 values for each finger M: number of fingers included,
    #                 kinova: kinova_states : (3,) - cartesian position of the arm end effector}
    def _get_one_representation(self, tactile_values, robot_states):
        if 'allegro' in self.representation_types or 'kinova' in self.representation_types:
            states = np.concatenate([robot_states[repr_type] for repr_type in self.representation_types if repr_type != 'tactile'], axis=0)
        if 'tactile' in self.representation_types:
            if self.use_encoder:
                tactile_image = self._get_tactile_image(tactile_values).unsqueeze(dim=0)
                tactile_repr = self.encoder(tactile_image)
                tactile_repr = tactile_repr.detach().cpu().numpy().squeeze()
            else:
                tactile_repr = tactile_values.flatten()

            if len(self.representation_types) == 1:
                return tactile_repr
            
            return np.concatenate((tactile_repr, states), axis=0)
            
        return states

    def _get_all_representations(
        self
    ):  
        print('Getting all representations')
        # For each tactile value and allegro tip position 
        # get one representation and add it to all representations
        repr_dim = 0
        if 'tactile' in self.representation_types:
            if self.use_encoder:
                repr_dim = self.cfg.encoder.out_dim
            else:
                repr_dim = len(self.sensor_indices) * 16 * 3
        if 'allegro' in self.representation_types:  repr_dim += len(self.allegro_finger_indices)
        if 'kinova' in self.representation_types: repr_dim += 7

        self.all_representations = np.zeros((
            len(self.tactile_indices), repr_dim
        ))

        print('all_representations.shape: {}'.format(self.all_representations.shape))
        pbar = tqdm(total=len(self.tactile_indices))

        for index in range(len(self.tactile_indices)):
            demo_id, tactile_id = self.tactile_indices[index]
            _, allegro_tip_id = self.allegro_indices[index]
            _, kinova_id = self.kinova_indices[index]

            # print('Found demo_id: {}'.format(demo_id))

            tactile_value = self.tactile_values[demo_id][tactile_id] # This should be (N,16,3)
            allegro_tip_position = self.allegro_tip_positions[demo_id][allegro_tip_id] # This should be (M*3,)
            kinova_state = self.kinova_states[demo_id][kinova_id]
            
            robot_states = dict(
                allegro = allegro_tip_position,
                kinova = kinova_state
            )
            representation = self._get_one_representation(
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
        demo_id, action_id = self.allegro_action_indices[self.state_id] 
        allegro_action = self.allegro_actions[demo_id][action_id] # Get the next commanded action (commanded actions are saved in that timestamp)
        if self.set_thumb_values is not None:
            allegro_action[-4:] = self.set_thumb_values
        action = dict(
            allegro = allegro_action
        )
        
        if 'kinova' in self.robots:
            _, kinova_id = self.kinova_indices[self.state_id] 
            kinova_action = self.kinova_states[demo_id][kinova_id] # Get the next saved kinova_state
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
        
        if 'kinova' in self.robots:
            kinova_cart_state = recv_robot_state['kinova']

        # Get the tactile image from the tactile values
        curr_tactile_values = tactile_values[self.sensor_indices,:,:]
        curr_fingertip_position = fingertip_positions[self.allegro_finger_indices]

        # print('curr_tactile_values.shape: {}, curr_fingertip_position.shape: {}, curr_kinova_state.shape: {}'.format(
        #     curr_tactile_values.shape, curr_fingertip_position.shape, kinova_cart_state.shape
        # ))

        assert curr_tactile_values.shape == (len(self.sensor_indices),16,3) and curr_fingertip_position.shape == (len(self.allegro_finger_indices),)

        curr_robot_state = dict(
            allegro = curr_fingertip_position,
            kinova = kinova_cart_state
        )
        # Get the representation with the given tactile value
        curr_representation = self._get_one_representation(
            curr_tactile_values, 
            curr_robot_state
        )
        _, nn_idxs = self.knn.get_k_nearest_neighbors(curr_representation, k=self.nn_k)
        # print('nn_idxs: {}'.format(nn_idxs))

        # Choose the action with the buffer 
        id_of_nn = self.buffer.choose(nn_idxs)
        nn_id = nn_idxs[id_of_nn]
        # print('chosen nn_id: {}'.format(nn_id))
        demo_id, action_id = self.allegro_action_indices[nn_id] 
        print('STATE ID: {}, CHOSEN DEMO ID: {}, ACTION ID: {}'.format(self.state_id, demo_id, action_id))
        nn_allegro_action = self.allegro_actions[demo_id][action_id] # Get the next commanded action (commanded actions are saved in that timestamp)
        if self.set_thumb_values is not None:
            nn_allegro_action[-4:] = self.set_thumb_values
        nn_action = dict(
            allegro = nn_allegro_action
        )
        
        if 'kinova' in self.robots:
            _, kinova_id = self.kinova_indices[nn_id] 
            nn_kinova_action = self.kinova_states[demo_id][kinova_id] # Get the next saved kinova_state
            nn_action['kinova'] = nn_kinova_action

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
        demo_id, tactile_id = self.tactile_indices[nn_id]
        _, allegro_tip_id = self.allegro_indices[nn_id]
        _, kinova_id = self.kinova_indices[nn_id]

        knn_tactile_values = self.tactile_values[demo_id][tactile_id]
        knn_tip_pos = self.allegro_tip_positions[demo_id][allegro_tip_id]
        knn_cart_pos = self.kinova_states[demo_id][kinova_id][:3]


        # Dump all the current state, nn state and curr image
        dump_camera_image()
        dump_whole_state(curr_tactile_values, curr_fingertip_position, curr_kinova_cart_pos, title='curr_state')
        dump_whole_state(knn_tactile_values, knn_tip_pos, knn_cart_pos, title='knn_state')

        # Plot from the dumped images
        dump_knn_state(
            dump_dir = os.path.join(self.out_dir, f'runs/run_{self.run_num}_ue_{self.use_encoder}'),
            img_name = 'state_{}.png'.format(str(self.state_id).zfill(2))
        ) # It will read the written images

    def _get_sorted_idxs(self, representation):
        l1_distances = self.all_representations - representation
        print('l1_distances.shape: {}'.format(l1_distances.shape))
        l2_distances = np.linalg.norm(l1_distances, axis = 1)

        sorted_idxs = np.argsort(l2_distances)
        return sorted_idxs

    def _get_knn_idxs(self, representation, k=0):
        sorted_idxs = self._get_sorted_idxs(representation)
        
        knn_idxs = sorted_idxs[:k+1]
        return knn_idxs

    # Get all the actions with the given indices - will be used in the
    # NN buffer
    def _get_actions_with_idxs(self, nn_idxs):
        actions = np.zeros((
            len(nn_idxs),
            len(self.allegro_actions[0][0]) # 16 is the action size
        ))

        for i in range(len(nn_idxs)):
            demo_id, action_id = self.allegro_action_indices[nn_idxs[i]]
            actions[i, :] = self.allegro_actions[demo_id][action_id]

        return actions

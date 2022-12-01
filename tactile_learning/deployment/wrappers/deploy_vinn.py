# Helper script to load models
import glob
import h5py
import numpy as np
import os
import pickle 
import torch
import torch.utils.data as data 
import torchvision.transforms as T

from collections import OrderedDict
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm 
from torch.nn.parallel import DistributedDataParallel as DDP

from holobot.robot.allegro.allegro_kdl import AllegroKDL
from tactile_learning.deployment.load_models import load_model
from tactile_learning.models.custom import TactileJointLinear, TactileImageEncoder

class NearestNeighborBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.exempted_queue = []

    def put(self, item):
        self.exempted_queue.append(item)
        if len(self.exempted_queue) > self.buffer_size:
            self.exempted_queue.pop(0)

    def get(self):
        item = self.exempted_queue[0]
        self.exempted_queue.pop(0)
        return item

    def choose(self, nn_idxs):
        for idx in range(len(nn_idxs)):
            # print('chosen_actions[idx]: {}'.format(chosen_actions[idx]))
            if nn_idxs[idx] not in self.exempted_queue:
                self.put(nn_idxs[idx])
                return idx

        return len(nn_idxs) - 1

class DeployVINN:
    def __init__(
        self,
        out_dir,
        sensor_indices = (3,7),
        allegro_finger_indices = (0,1)
    ):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"

        torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
        torch.cuda.set_device(0)

        self.sensor_indices = sensor_indices 
        self.allegro_finger_indices = [j for i in allegro_finger_indices for j in range(i*3,(i+1)*3)]

        self.device = torch.device('cuda:0')
        self.cfg = OmegaConf.load(os.path.join(out_dir, '.hydra/config.yaml'))
        self.data_path = self.cfg.data_dir
        model_path = os.path.join(out_dir, 'models/byol_encoder.pt')

        self.encoder = load_model(self.cfg, self.device, model_path)
        self.encoder.eval() 

        self.resize_transform = T.Resize((8, 8))

        tactile_values, allegro_tip_positions = self._load_data()
        self._get_all_representations(tactile_values, allegro_tip_positions)

        self.kdl_solver = AllegroKDL()
        self.last_action = None
        self.buffer = NearestNeighborBuffer(20)

    def _load_data(self):
        roots = glob.glob(f'{self.data_path}/demonstration_*')
        roots = sorted(roots)

        self.tactile_indices = [] 
        self.allegro_indices = [] 
        self.allegro_action_indices = [] 
        self.allegro_actions = [] 
        tactile_values = [] 
        allegro_tip_positions = []

        for root in roots:
            # Load the indices
            with open(os.path.join(root, 'tactile_indices.pkl'), 'rb') as f:
                self.tactile_indices += pickle.load(f)
            with open(os.path.join(root, 'allegro_indices.pkl'), 'rb') as f:
                self.allegro_indices += pickle.load(f)
            with open(os.path.join(root, 'allegro_action_indices.pkl'), 'rb') as f:
                self.allegro_action_indices += pickle.load(f)

            # Load the data
            with h5py.File(os.path.join(root, 'allegro_fingertip_states.h5'), 'r') as f:
                # print(f['positions'][()].shape)
                allegro_tip_positions.append(f['positions'][()][:, self.allegro_finger_indices])
            with h5py.File(os.path.join(root, 'allegro_commanded_joint_states.h5'), 'r') as f:
                self.allegro_actions.append(f['positions'][()]) # Positions are to be learned - since this is a position control
            with h5py.File(os.path.join(root, 'touch_sensor_values.h5'), 'r') as f:
                tactile_values.append(f['sensor_values'][()][:,self.sensor_indices,:,:])

        # print(self.allegro_tip_positions[0].shape, self.tactile_values[0].shape)
        return tactile_values, allegro_tip_positions

    def _get_tactile_image(self, tactile_value):
        tactile_image = torch.FloatTensor(tactile_value)
        tactile_image = tactile_image.reshape((
            len(self.sensor_indices),  # Total number of sensors - (2,16,3)
            4, 
            4,
            -1
        ))
        # TODO: This will only work for this grid
        tactile_image = torch.concat((tactile_image[0], tactile_image[1]), dim=1)
        tactile_image = torch.permute(tactile_image, (2,0,1))

        return self.resize_transform(tactile_image)

    # tactile_values: (2,16,3)
    # allegro_tip_positions: (6,) - 3 values for each finger
    def _get_one_representation(self, tactile_values, allegro_tip_positions):
        # For each tactile value get the tactile image
        tactile_image = self._get_tactile_image(tactile_values).unsqueeze(dim=0)
        # print('tactile_image.shape: {}'.format(tactile_image.shape))
        tactile_repr = self.encoder(tactile_image)
        tactile_repr = tactile_repr.detach().cpu().numpy().squeeze() # Remove the axes with dimension 1
        # print('tactile_repr.shape: {}'.format(tactile_repr.shape))
        # It should be (64,)
        return np.concatenate((tactile_repr, allegro_tip_positions), axis=0)


    def _get_all_representations(
        self,
        tactile_values,
        allegro_tip_positions
    ):  
        print('Getting all representations')
        pbar = tqdm(total=len(self.tactile_indices))
        # For each tactile value and allegro tip position 
        # get one representation and add it to all representations
        repr_dim = self.cfg.encoder.out_dim + len(self.cfg.dataset.allegro_finger_indices) * 3 
        # print('repr_dim: {}'.format(repr_dim))
        self.all_representations = np.zeros((
            len(self.tactile_indices), repr_dim
        ))

        for index in range(len(self.tactile_indices)):
            demo_id, tactile_id = self.tactile_indices[index]
            _, allegro_tip_id = self.allegro_action_indices[index]

            tactile_value = tactile_values[demo_id][tactile_id] # This should be (2,16,3)
            # print('tactile_value.shape: {}'.format(tactile_value.shape))
            allegro_tip_position = allegro_tip_positions[demo_id][tactile_id] # This should be (6,)
            # print('allegro_tip_position.shape: {}'.format(allegro_tip_position.shape))
            representation = self._get_one_representation(
                tactile_value, 
                allegro_tip_position
            )
            # Add only tip positions as the representation
            self.all_representations[index, -6:] = representation[-6:] # TODO: Fix this
            # self.all_representations[index,  :] = representation[:]
            # print(f'last repr: {self.all_representations[index]}')
            pbar.update(1)

        pbar.close()

    # tactile_values.shape: (16,15,3)
    # joint_state.shape: (16)
    def get_action(self, tactile_values, joint_state):
        # Get the allegro tip positions with kdl solver 
        fingertip_positions = self.kdl_solver.get_fingertip_coords(joint_state) # - fingertip position.shape: (12)

        # Get the tactile image from the tactile values
        curr_tactile_values = tactile_values[self.sensor_indices,:,:]
        curr_fingertip_position = fingertip_positions[self.allegro_finger_indices]

        print('curr_tactile_values.shape: {}, curr_fingertip_position.shape: {}'.format(
            curr_tactile_values.shape, curr_fingertip_position.shape
        ))

        assert curr_tactile_values.shape == (2,16,3) and curr_fingertip_position.shape == (6,)

        # Get the representation with the given tactile value
        curr_representation = self._get_one_representation(
            curr_tactile_values, 
            curr_fingertip_position
        )

        new_curr_representation = np.zeros_like(curr_representation)
        new_curr_representation[-6:] = curr_representation[-6:]
        # print('new_curr_repr: {}'.format(new_curr_representation))

        k = 10
        nn_idxs = self._get_knn_idxs(new_curr_representation, k=k) # TODO: Fix this
        print('nn_idxs: {}'.format(nn_idxs))

        # Choose the action with the buffer 
        # nn_actions = self._get_actions_with_idxs(nn_idxs)
        # print('nn_actions.shape: {}'.format(nn_actions.shape))
        id_of_nn = self.buffer.choose(nn_idxs)
        nn_id = nn_idxs[id_of_nn]
        print('chosen nn_id: {}'.format(nn_id))
        demo_id, action_id = self.allegro_action_indices[nn_id]
        nn_action = self.allegro_actions[demo_id][action_id]

        # Get the applied action at that id
        # Traverse through actions and find the action that is somewhat distance
        # from the last action
        # for i in range(k):
        #     demo_id, action_id = self.allegro_action_indices[nn_idxs[i]]
        #     nn_action = self.allegro_actions[demo_id][action_id]
        #     if self.last_action is None: 
        #         self.last_action = nn_action 
        #         break
        #     else:
        #         l2_distance = np.linalg.norm(self.last_action - nn_action)
        #         print(f'i: {i} - l2 distance between commanded joint positions: {l2_distance}')
        #         if l2_distance > 0.01:
        #             self.last_action = nn_action
        #             break

        print('nn_action: {}'.format(nn_action))

        return nn_action

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

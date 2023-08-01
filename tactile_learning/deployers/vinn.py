# Helper script to load models
import cv2
import glob
import numpy as np
import os
import pickle
import torch
import torchvision.transforms as T


from omegaconf import OmegaConf
from tqdm import tqdm 

from holobot.constants import *

from holobot.robot.allegro.allegro_kdl import AllegroKDL

from tactile_learning.models import *
from tactile_learning.tactile_data import *
from tactile_learning.utils import *

from .deployer import Deployer
from .utils.nn_buffer import NearestNeighborBuffer

class VINN(Deployer):
    def __init__(
        self,
        data_path,
        deployment_dump_dir,
        tactile_out_dir=None, # If these are None then it's considered we'll use non trained encoders
        image_out_dir=None,
        image_model_type = 'byol',
        representation_types = ['image', 'tactile', 'kinova', 'allegro', 'torque'], # Torque could be used
        representation_importance = [1,1,1,1], 
        tactile_repr_type = 'tdex', # raw, shared, stacked, tdex, sumpool, pca (uses the encoder passed)
        tactile_shuffle_type = None,   
        nn_buffer_size=100,
        nn_k=20,
        demos_to_use=[0],
        view_num = 0, # View number to use for image
        open_loop = False, # Open loop vinn means that we'll run the demo after getting the first frame from KNN
        dump_deployment_info = False
    ):
        
        self.set_up_env() 

        self.representation_types = representation_types
        self.demos_to_use = demos_to_use

        self.device = torch.device('cuda:0')
        self.view_num = view_num
        self.open_loop = open_loop

        tactile_cfg, tactile_encoder, _ = self._init_encoder_info(self.device, tactile_out_dir, 'tactile')
        self.tactile_img = TactileImage(
            tactile_image_size = tactile_cfg.tactile_image_size, 
            shuffle_type = tactile_shuffle_type
        )
        self.tactile_repr = TactileRepresentation(
            encoder_out_dim = tactile_cfg.encoder.out_dim,
            tactile_encoder = tactile_encoder,
            tactile_image = self.tactile_img,
            representation_type = tactile_repr_type
        )

        # self.image_cfg, self.image_encoder, self.image_transform = self._init_encoder_info(self.device, image_out_dir, 'image')
        self.image_cfg, self.image_encoder, self.image_transform = init_encoder_info(
            device = self.device,
            out_dir = image_out_dir,
            view_num = view_num, 
            model_type = image_model_type,
            encoder_type = 'image'
        )
        self.inv_image_transform = get_inverse_image_norm()

        self.roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
        self.data_path = data_path
        self.data = load_data(self.roots, demos_to_use=demos_to_use) # This will return all the desired indices and the values

        self._get_all_representations()
        self.state_id = 0 # Increase it with each get_action

        self.nn_k = nn_k
        self.kdl_solver = AllegroKDL()
        self.buffer = NearestNeighborBuffer(nn_buffer_size)
        self.knn = ScaledKNearestNeighbors(
            self.all_representations, # Both the input and the output of the nearest neighbors are
            self.all_representations,
            representation_types,
            representation_importance,
            self.tactile_repr.size
        )

        self.deployment_dump_dir = deployment_dump_dir
        if dump_deployment_info:
            os.makedirs(self.deployment_dump_dir, exist_ok=True)
        self.dump_deployment_info = dump_deployment_info
        if dump_deployment_info:
            self.deployment_info = dict(
                all_representations = self.all_representations,
                curr_representations = [], # representations will be appended to this list
                closest_representations = [],
                neighbor_ids = [],
                images = [], 
                tactile_values = []
            )

        self.visualization_info = dict(
            state_ids = [],
            images = [], 
            tactile_values = [],
            id_of_nns = [], 
            nn_idxs = [], 
            nn_dists = []
        )

    
    def _init_encoder_info(self, device, out_dir, encoder_type='tactile'): # encoder_type: either image or tactile
        if encoder_type == 'tactile' and  out_dir is None:
            encoder = alexnet(pretrained=True, out_dim=512, remove_last_layer=True)
            cfg = OmegaConf.create({'encoder':{'out_dim':512}, 'tactile_image_size':224})
        
        elif encoder_type =='image' and out_dir is None: # Load the pretrained encoder 
            encoder = resnet18(pretrained=True, out_dim=512) # These values are set
            cfg = OmegaConf.create({"encoder":{"out_dim":512}})
        
        else:
            cfg = OmegaConf.load(os.path.join(out_dir, '.hydra/config.yaml'))
            model_path = os.path.join(out_dir, 'models/byol_encoder_best.pt')
            encoder = load_model(cfg, device, model_path)
        encoder.eval() 

        if encoder_type == 'image':
            transform = T.Compose([
                T.Resize((480,640)),
                T.Lambda(self._crop_transform),
                T.Resize(480),
                T.ToTensor(),
                T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS),
            ]) 
        else:
            transform = None # This is separately set for tactile

        return cfg, encoder, transform
    
    def _crop_transform(self, image):
        return crop_transform(image, camera_view=self.view_num)

    def _load_dataset_image(self, demo_id, image_id):
        dset_img = load_dataset_image(self.data_path, demo_id, image_id, self.view_num)
        img = self.image_transform(dset_img)
        return torch.FloatTensor(img) 
    
    # tactile_values: (N,16,3) - N: Number of sensors
    # robot_states: { allegro: allegro_tip_positions: 12 - 3*4, End effector cartesian position for each finger tip
    #                 kinova: kinova_states : (3,) - Cartesian position of the arm end effector}
    def _get_one_representation(self, image, tactile_values, robot_states):
        for i,repr_type in enumerate(self.representation_types):
            if repr_type == 'allegro' or repr_type == 'kinova' or repr_type == 'torque':
                new_repr = robot_states[repr_type] # These could be received directly from the robot states
            elif repr_type == 'tactile':
                # tactile_values = torch.FloatTensor(tactile_values).to(self.device)
                new_repr = self.tactile_repr.get(tactile_values)
            elif repr_type == 'image':
                new_repr = self.image_encoder(image.unsqueeze(dim=0).to(self.device)) # Add a dimension to the first axis so that it could be considered as a batch
                new_repr = new_repr.detach().cpu().numpy().squeeze()

            if i == 0:
                curr_repr = new_repr 
            else: 
                curr_repr = np.concatenate([curr_repr, new_repr], axis=0)
                
        return curr_repr
    
    def _get_all_representations(self):
        print('Getting all representations')

        self.all_representations = []

        pbar = tqdm(total=len(self.data['tactile']['indices']))
        for index in range(len(self.data['tactile']['indices'])):
            # Get the representation data
            repr_data = self._get_data_with_id(index, visualize=False)

            representation = self._get_one_representation(
                repr_data['image'],
                repr_data['tactile_value'], 
                repr_data['robot_states'] 
            )
            self.all_representations.append(representation)
            pbar.update(1)

        pbar.close()

        self.all_representations = np.stack(self.all_representations, axis=0)

    def save_deployment(self):
        if self.dump_deployment_info:
            with open(os.path.join(self.deployment_dump_dir, 'deployment_info.pkl'), 'wb') as f:
                pickle.dump(self.deployment_info, f)

        with open(os.path.join(self.deployment_dump_dir, 'visualization_info.pkl'), 'wb') as f:
            pickle.dump(self.visualization_info, f)

        # Visualize
        if len(self.visualization_info['state_ids']) > 0: # It was asked to visualize
            pbar = tqdm(total=len(self.visualization_info['state_ids']))
            for i,state_id in enumerate(self.visualization_info['state_ids']):

                # print('')
                self._visualize_state(
                    curr_tactile_values = self.visualization_info['tactile_values'][i],
                    curr_fingertip_position = None, # We are not dumping state for now
                    curr_kinova_cart_pos = None,
                    id_of_nn = self.visualization_info['id_of_nns'][i],
                    nn_idxs = self.visualization_info['nn_idxs'][i],
                    nn_separate_dists = self.visualization_info['nn_dists'][i],
                    image = self.visualization_info['images'][i], 
                    state_id = state_id
                )
                pbar.set_description(f'Dumping visualization state: {state_id}')
                pbar.update(1)

            pbar.close()

    def get_action(self, tactile_values, recv_robot_state, visualize=False):
        if self.open_loop:
            if self.state_id == 0: # Get the closest nearest neighbor id for the first state
                action, self.open_loop_start_id = self._get_knn_action(tactile_values, recv_robot_state, visualize)
            else:
                action = self._get_open_loop_action(tactile_values, visualize)
        else:
            action = self._get_knn_action(tactile_values, recv_robot_state, visualize)
        
        print(f'STATE ID: {self.state_id}')
        return  action
    
    def _get_open_loop_action(self, tactile_values, visualize):
        demo_id, action_id = self.data['allegro_actions']['indices'][self.state_id+self.open_loop_start_id] 
        allegro_action = self.data['allegro_actions']['values'][demo_id][action_id] # Get the next commanded action (commanded actions are saved in that timestamp)
        
        demo_id, allegro_state_id = self.data['allegro_joint_states']['indices'][self.state_id+self.open_loop_start_id] 
        allegro_state = self.data['allegro_joint_states']['values'][demo_id][allegro_state_id]
        allegro_action[-4:] = allegro_state[-4:] # Fix the thumb
        action = dict(
            allegro = allegro_action
        )
        
        _, kinova_id = self.data['kinova']['indices'][self.state_id+self.open_loop_start_id] 
        kinova_action = self.data['kinova']['values'][demo_id][kinova_id] # Get the next saved kinova_state
        action['kinova'] = kinova_action

        if visualize: 
            image = self._get_curr_image()
            curr_image = self.inv_image_transform(image).numpy().transpose(1,2,0)
            curr_image_cv2 = cv2.cvtColor(curr_image*255, cv2.COLOR_RGB2BGR)

            tactile_image = self._get_tactile_image_for_visualization(tactile_values)
            dump_whole_state(tactile_values, tactile_image, None, None, title='curr_state', vision_state=curr_image_cv2)
            curr_state = cv2.imread('curr_state.png')
            image_path = os.path.join(self.deployment_dump_dir, f'state_{str(self.state_id).zfill(2)}.png')
            cv2.imwrite(image_path, curr_state)

        self.state_id += 1
        
        return action 
    
    # tactile_values.shape: (16,15,3)
    # robot_state: {allegro: allegro_joint_state (16,), kinova: kinova_cart_state (3,)}
    def _get_knn_action(self, curr_tactile_values, recv_robot_state, visualize=False):
        # Get the current state of the robot
        allegro_joint_state = recv_robot_state['allegro']
        fingertip_positions = self.kdl_solver.get_fingertip_coords(allegro_joint_state) # - fingertip position.shape: (12)
        kinova_cart_state = recv_robot_state['kinova']
        allegro_joint_torque = recv_robot_state['torque']
        curr_robot_state = dict(
            allegro = fingertip_positions,
            kinova = kinova_cart_state,
            torque = allegro_joint_torque
        )

        # Get the current visual image
        image = self._get_curr_image()

        # Get the representation with the given tactile value
        curr_representation = self._get_one_representation(
            image,
            curr_tactile_values, 
            curr_robot_state
        )


        # Choose the action with the buffer 
        _, nn_idxs, nn_separate_dists = self.knn.get_k_nearest_neighbors(curr_representation, k=self.nn_k)
        id_of_nn = self.buffer.choose(nn_idxs)
        nn_id = nn_idxs[id_of_nn]
        if nn_id+1 >= len(self.data['allegro_actions']['indices']): # If the chosen action is the action after the last action
            nn_idxs = np.delete(nn_idxs, id_of_nn)
            id_of_nn = self.buffer.choose(nn_idxs)
            nn_id = nn_idxs[id_of_nn]

        demo_id, action_id = self.data['allegro_actions']['indices'][nn_id+1]  # Get the next commanded action (commanded actions are saved in that timestamp)
        nn_allegro_action = self.data['allegro_actions']['values'][demo_id][action_id]

        demo_id, allegro_state_id = self.data['allegro_joint_states']['indices'][nn_id+1] 
        nn_allegro_state = self.data['allegro_joint_states']['values'][demo_id][allegro_state_id]
        nn_allegro_action[-4:] = nn_allegro_state[-4:] # Set the thumb to the state rather than the action since we fix the thumb in the demonstrations
        nn_action = dict(
            allegro = nn_allegro_action
        )
        
        _, kinova_id = self.data['kinova']['indices'][nn_id+1] # Get the next kinova state (which is for kinova robot the same as the next commanded action)
        nn_kinova_action = self.data['kinova']['values'][demo_id][kinova_id]
        nn_action['kinova'] = nn_kinova_action

        # Save everything to deployment_info
        if self.dump_deployment_info:
            self.deployment_info['curr_representations'].append(curr_representation)
            self.deployment_info['images'].append(image)
            self.deployment_info['tactile_values'].append(curr_tactile_values)
            self.deployment_info['neighbor_ids'].append(nn_idxs[0])
            closest_representation = self.all_representations[nn_idxs[0]]
            self.deployment_info['closest_representations'].append(closest_representation)

        # Visualize if given 
        if visualize: 
            self.visualization_info['state_ids'].append(self.state_id)
            self.visualization_info['images'].append(image)
            self.visualization_info['tactile_values'].append(curr_tactile_values)
            self.visualization_info['id_of_nns'].append(id_of_nn)
            self.visualization_info['nn_idxs'].append(nn_idxs)
            self.visualization_info['nn_dists'].append(nn_separate_dists)

        self.state_id += 1

        if self.open_loop:
            return nn_action, nn_id

        return nn_action

    def _visualize_state(self, curr_tactile_values, curr_fingertip_position, curr_kinova_cart_pos, id_of_nn, nn_idxs, nn_separate_dists, state_id=None, image=None):
        # Get the current image 
        if image is None:
            image = self._get_curr_image()
        curr_image = self.inv_image_transform(image).numpy().transpose(1,2,0)
        curr_image_cv2 = cv2.cvtColor(curr_image*255, cv2.COLOR_RGB2BGR)
        curr_tactile_image = self.tactile_img.get_tactile_image_for_visualization(curr_tactile_values) 

        nn_id = nn_idxs[id_of_nn]
        # Get the next visualization data
        knn_vis_data = self._get_data_with_id(nn_id, visualize=True)

        # Get the demo ids of the closest 3 neighbor
        demo_ids = []
        demo_nums = []
        viz_id_of_nns = []
        for i in range(3):
            if not (id_of_nn+i >= len(nn_idxs)):
                viz_nn_id = nn_idxs[id_of_nn+i]
                viz_id_of_nns.append(id_of_nn+i)
            else:
                viz_id_of_nns.append(viz_id_of_nns[-1])
            demo_id, _ = self.data['tactile']['indices'][viz_nn_id]
            demo_ids.append(demo_id)
            demo_nums.append(int(self.roots[demo_id].split('/')[-1].split('_')[-1]))
    
        dump_whole_state(curr_tactile_values, curr_tactile_image, curr_fingertip_position, curr_kinova_cart_pos, title='curr_state', vision_state=curr_image_cv2)
        dump_whole_state(knn_vis_data['tactile_values'], knn_vis_data['tactile_image'], knn_vis_data['allegro'], knn_vis_data['kinova'], title='knn_state', vision_state=knn_vis_data['image'])
        dump_repr_effects(nn_separate_dists, viz_id_of_nns, demo_nums, self.representation_types)
        if state_id is None:
            state_id = self.state_id
        dump_knn_state(
            dump_dir = self.deployment_dump_dir,
            img_name = 'state_{}.png'.format(str(state_id).zfill(2)),
            image_repr = True,
            add_repr_effects = True,
            include_temporal_states = False
        )

    def _get_data_with_id(self, id, visualize=False):
        demo_id, tactile_id = self.data['tactile']['indices'][id]
        _, allegro_tip_id = self.data['allegro_tip_states']['indices'][id]
        _, kinova_id = self.data['kinova']['indices'][id]
        _, image_id = self.data['image']['indices'][id]
        _, allegro_state_id = self.data['allegro_joint_states']['indices'][id]

        tactile_value = self.data['tactile']['values'][demo_id][tactile_id] # This should be (N,16,3)
        allegro_tip_position = self.data['allegro_tip_states']['values'][demo_id][allegro_tip_id] # This should be (M*3,)
        kinova_state = self.data['kinova']['values'][demo_id][kinova_id]
        image = self._load_dataset_image(demo_id, image_id)
        
        if visualize:
            tactile_image = self.tactile_img.get_tactile_image_for_visualization(tactile_value) 
            kinova_cart_pos = kinova_state[:3] # Only position is used
            vis_image = self.inv_image_transform(image).numpy().transpose(1,2,0)
            vis_image = cv2.cvtColor(vis_image*255, cv2.COLOR_RGB2BGR)

            visualization_data = dict(
                image = vis_image,
                kinova = kinova_cart_pos, 
                allegro = allegro_tip_position, 
                tactile_values = tactile_value,
                tactile_image = tactile_image
            )
            return visualization_data

        else:
            allegro_joint_torque = self.data['allegro_joint_states']['torques'][demo_id][allegro_state_id] # This is the torque to be used
            robot_states = dict(
                allegro = allegro_tip_position,
                kinova = kinova_state,
                torque = allegro_joint_torque
            )
            data = dict(
                image = image,
                tactile_value = tactile_value, 
                robot_states = robot_states
            )
            return data

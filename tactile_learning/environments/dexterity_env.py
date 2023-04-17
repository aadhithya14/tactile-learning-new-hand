# Main script for hand interractions 
import cv2 
import gym
import numpy as np
import os
import torch
import torchvision.transforms as T

from gym import spaces
from holobot_api import DeployAPI
from holobot.robot.allegro.allegro_kdl import AllegroKDL
from holobot.utils.network import ZMQCameraSubscriber
from PIL import Image as im

from tactile_learning.tactile_data import TactileImage, TactileRepresentation
from tactile_learning.models import init_encoder_info
from tactile_learning.utils import *

class DexterityEnv(gym.Env):
        def __init__(self, # We will use both the hand and the arm
            tactile_model_dir,
            host_address = "172.24.71.240",
            camera_num = 0
        ):
            # print(camera_num, "CAMERA_NUM")
            self.width = 224
            self.height = 224
            self.view_num = camera_num

            self.deploy_api = DeployAPI(
                host_address=host_address,
                required_data={"rgb_idxs": [camera_num], "depth_idxs": []}
            )

            self.home_state = dict(
                allegro = np.array([
                    -0.0658244726801581, 0.11152991296986751, 0.036465840916854717, 0.29693057660614736, # Index
                    -0.09053422635521813, 0.21657171862672447, -0.17754325611897262, 0.27011271061536507, # Middle
                    0.012094523852233988, 0.11196786731996372, -0.017784060790178313, 0.2670852707825862, # Ring
                    0.8499175389966154, 0.3062633015641964, 0.7989875369900138, 0.46722180902731736 # Thumb
                ]),
                kinova = np.array([-0.47642678,  0.27087545,  0.34734036, -0.06232093, -0.67618454, 0.05119989,  0.73230404])
            )

            self._robot = AllegroKDL()

            device = torch.device('cuda:0')
            tactile_cfg, tactile_encoder, _ = init_encoder_info(device, tactile_model_dir, 'tactile')
            tactile_img = TactileImage(
                tactile_image_size = tactile_cfg.tactile_image_size, 
                shuffle_type = None
            )
            self.tactile_repr = TactileRepresentation(
                encoder_out_dim = tactile_cfg.encoder.out_dim,
                tactile_encoder = tactile_encoder,
                tactile_image = tactile_img,
                representation_type = 'tdex'
            )

            self.image_subscriber = ZMQCameraSubscriber(
                host = host_address,
                port = 10005 + self.view_num,
                topic_type = 'RGB'
            )
            self.image_transform = T.Compose([
                T.Resize((480,640)),
                T.Lambda(self._crop_transform),
                T.Resize((self.height, self.width))
            ]) 

        def set_up_env(self):
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29505"

            torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
            torch.cuda.set_device(0)

        def _get_curr_image(self):
            image, _ = self.image_subscriber.recv_rgb_image()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = im.fromarray(image)
            img = self.image_transform(image)
            return img

        def _crop_transform(self, image):
            return crop_transform(image, camera_view=self.view_num)

        def init_hand(self):
            # while True: 
            #     try:
            #         self.deploy_api.send_robot_action(self.home_state)
            #         break 
            #     except:
            #         print('Error in init_hand') # NOTE: delete this weird 
            #         pass
            self.deploy_api.send_robot_action(self.home_state)

        def step(self, action):
            print('action.shape: {}'.format(action.shape))
            action *= (1.0/5) 
            try: 
                hand_joint_action = self._robot.get_joint_state_from_coord(
                    action[0:3], action[3:6], action[6:9], action[9:12],
                    self.deploy_api.get_robot_state()['allegro']['position'])
                self.deploy_api.send_robot_action({
                    'allegro': hand_joint_action, 
                    'kinova':  action[12:]
                })
                # self.hand.send_robot_action({self.robot_name: converted_action})
            except:
                print("IK error")
            
            # Get the observations
            obs = {}
            obs['features'] = self.deploy_api.get_robot_state() # NOTE: having the features should be better and faster as well
            obs['pixels'] = self._get_curr_image()
            # obs['depth'] = self.hand.get_depth_images() - NOTE we're not using depth for now
            
            sensor_state = self.deploy_api.get_sensor_state()
            tactile_values = sensor_state['xela']['sensor_values']
            obs['tactile'] = self.tactile_repr.get(tactile_values)

            return obs, 0, False, {'is_success': False} #obs, reward, done, infos


        def render(self, mode='rbg_array', width=0, height=0):
            return self._get_curr_image()
    
        def reset(self): 
            self.init_hand()
            obs = {}
            obs['features'] = self.deploy_api.get_robot_state() # NOTE: having the features should be better and faster as well
            obs['pixels'] = self._get_curr_image()
            
            sensor_state = self.deploy_api.get_sensor_state()
            tactile_values = sensor_state['xela']['sensor_values']
            obs['tactile'] = self.tactile_repr.get(tactile_values)
            return obs


        def get_reward():
            pass
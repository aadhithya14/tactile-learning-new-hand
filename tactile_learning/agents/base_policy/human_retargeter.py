
import glob


# Holobot imports
from copy import deepcopy as copy
from shapely.geometry import Point, Polygon 
from shapely.ops import nearest_points
from holobot.components.operators.calibrators.allegro import OculusThumbBoundCalibrator
from holobot.robot.allegro.allegro import AllegroHand
from holobot.robot.allegro.allegro_retargeters import AllegroJointControl, AllegroKDLControl
from holobot.utils.files import *
from holobot.utils.timer import FrequencyTimer
from holobot.constants import *

# tactile_Learning imports
from tactile_learning.utils import load_human_data

from .base_policy import BasePolicy

class HumanRetargeter(BasePolicy):
    def __init__(
        self,
        expert_demos,
        expert_id,
        host = '172.24.71.240', # These values are set for this computer so will not be taking them as parameters 
        port = 8089, # This will not be used by the OculusTHumbBoundCalibrator anyways
        **kwargs
    ):

        self.expert_id = expert_id 
        self.set_expert_demos(expert_demos)

        # Get the host and the port
        self._host = host 
        self._port = port

        # Create the drivers
        self._robot = AllegroHand()
        self.finger_joint_solver = AllegroJointControl()
        self.fingertip_solver = AllegroKDLControl(debug_thumb_configs=None)

        # Initialzing the moving average queues
        self.moving_average_queues = {
            'thumb': [],
            'index': [],
            'middle': [],
            'ring': []
        }

        # Calibrating to get the thumb bounds
        self._calibrate_bounds()

        # Getting the bounds for the allegro hand
        allegro_bounds_path = get_path_in_package(
            '/home/irmak/Workspace/Holo-Bot/holobot/components/operators/configs/allegro.yaml'
        )
        self.allegro_bounds = get_yaml_data(allegro_bounds_path)

        self._timer = FrequencyTimer(VR_FREQ)

        self.thumb_angle_calculator = self._get_thumb_angles
        
    def _calibrate_bounds(self):
        # self.notify_component_start('calibration')
        calibrator = OculusThumbBoundCalibrator(self._host, self._port)
        self.hand_thumb_bounds = calibrator.get_bounds() # Provides [thumb-index bounds, index-middle bounds, middle-ring-bounds]

    def _get_thumb_angles(self, thumb_keypoints, curr_angles):
        # We will be using polygon implementations of shapely library to test this
        planar_point = Point(thumb_keypoints)
        planar_thumb_bounds = Polygon(self.hand_thumb_bounds[:4])

        # Get the closest point from the thumb to the point
        # this will return the point if it's inside the bounds
        closest_point = nearest_points(planar_thumb_bounds, planar_point)[0]
        closest_point_coords = [closest_point.x, closest_point.y, thumb_keypoints[2]]
        return self.fingertip_solver.thumb_motion_3D(
            hand_coordinates = closest_point_coords,
            xy_hand_bounds = self.hand_thumb_bounds[:4],
            yz_robot_bounds = self.allegro_bounds['thumb_bounds'][0]['projective_bounds'], # NOTE: We assume there is only one bound now
            z_hand_bound = self.hand_thumb_bounds[4],
            x_robot_bound = self.allegro_bounds['thumb_bounds'][0]['x_bounds'],
            moving_avg_arr = self.moving_average_queues['thumb'], 
            curr_angles = curr_angles
        )

    def _get_allegro_joint_angles(self, hand_keypoints):
        # hand_keypoints = self._get_finger_coords() # TODO 
        desired_joint_angles = copy(self._robot.get_joint_position())

        for finger_type in ['index', 'middle', 'ring', 'thumb']:
            if finger_type == 'thumb':
                desired_joint_angles = self.thumb_angle_calculator(
                    hand_keypoints['thumb'][-1],
                    desired_joint_angles
                )
            else:
                desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
                    finger_type = finger_type,
                    finger_joint_coords = hand_keypoints[finger_type],
                    curr_angles = desired_joint_angles,
                    moving_avg_arr = self.moving_average_queues[finger_type]
                )

        return desired_joint_angles
    
    def act(self, obs, episode_step, **kwargs):
        is_done = False 
        if episode_step >= len(self.expert_demos[self.expert_id]['keypoints']):
            episode_step = len(self.expert_demos[self.expert_id]['keypoints'])-1
            is_done = True

        # Get allegro joint angles
        raw_keypoints = self.expert_demos[self.expert_id]['keypoints'][episode_step]
        hand_keypoints = dict(
            index = np.vstack([raw_keypoints[0], raw_keypoints[OCULUS_JOINTS['index']]]),
            middle = np.vstack([raw_keypoints[0], raw_keypoints[OCULUS_JOINTS['middle']]]),
            ring = np.vstack([raw_keypoints[0], raw_keypoints[OCULUS_JOINTS['ring']]]),
            thumb =  np.vstack([raw_keypoints[0], raw_keypoints[OCULUS_JOINTS['thumb']]])
        )
        allegro_joint_angles = self._get_allegro_joint_angles(hand_keypoints)

        # Get kinova joint angles
        kinova_action = obs['features'][-7:]
        action = np.concatenate([allegro_joint_angles, kinova_action], axis=-1)

        print('ACTION IN BASE ACT: {}'.format(action))

        return action, is_done
    

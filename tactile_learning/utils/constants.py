# Alexnet means and stds
TACTILE_IMAGE_MEANS = [0.485, 0.456, 0.406]
TACTILE_IMAGE_STDS = [0.229, 0.224, 0.225]

# These constants are used to clamp 
TACTILE_PLAY_DATA_CLAMP_MIN = -1000
TACTILE_PLAY_DATA_CLAMP_MAX = 1000

# Constans for camera images
VISION_IMAGE_MEANS = [0.4191, 0.4445, 0.4409]
VISION_IMAGE_STDS = [0.2108, 0.1882, 0.1835]

# Robotic constants
KINOVA_CARTESIAN_POS_SIZE = 7 
ALLEGRO_JOINT_NUM = 16
ALLEGRO_EE_REPR_SIZE = 12 

# Play data path - should be changed 
PLAY_DATA_PATH = '/data/tactile_learning/play_data'

# Constants for the IBC training
THUMB_JOINT_LIMITS = [
    [0,1.5], [-0.1,1], [-0.1,1.3], [-0.1,1.3]
]
FINGER_JOINT_LIMITS = [
    [-0.2, 0.2],[-0.1, 1.3],[-0.1,1.3],[-0.1,1.3]
]
KINOVA_JOINT_LIMITS = [
    [-0.48, -0.55],[0.15,0.25], [0.3,0.45], # X,Y,Z
    [-0.1,0.1], [-0.7,-0.5],[0.0,0.1],[0.6,0.8] 
]
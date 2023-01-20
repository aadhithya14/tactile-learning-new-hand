## Joystick task ##
# TACTILE_IMAGE_MEANS = [
#     4.32517252e-05,
#     -4.31482843e-05,
#     5.41983846e-04
# ] 

# TACTILE_IMAGE_STDS = [
#     0.06121121,
#     0.06255601,
#     0.08893731
# ]
# TACTILE_IMAGE_MEANS = [
#     -1.9989,
#     0.9851,
#     0.0650
# ]
# TACTILE_IMAGE_STDS = [
#     60.2708,
#     39.4647,
#     41.2752
# ]

# Play Data Tactile Mean and Stds - The Whole Hand 
# TACTILE_IMAGE_MEANS = [-6.5651,  3.4824, 10.0751]
# TACTILE_IMAGE_STDS = [183.9877, 151.5086, 128.7941]

# Stats for the play data
# PLAY_DATA_TACTILE_MEAN = torch.Tensor([-3.4821,  0.8840, 11.0220])
# PLAY_DATA_TACTILE_STD = torch.Tensor([96.1595, 67.3064, 72.4123])
# PLAY_DATA_TACTILE_MIN = torch.Tensor([ -5.9464, -10.1432,  -9.6585]).unsqueeze(1).unsqueeze(1)
# PLAY_DATA_TACTILE_MAX = torch.Tensor([ 5.9450, 10.1289,  9.4191]).unsqueeze(1).unsqueeze(1)

# # Stats for gamepad task
# GAMEPAD_TACTILE_MEAN = torch.Tensor([-12.8636,  -1.1549,   1.3344])
# GAMEPAD_TACTILE_STD = torch.Tensor([66.7323, 43.5684, 49.4381])
# GAMEPAD_TACTILE_MIN = torch.Tensor([-14.7925, -22.9259, -20.2543]).unsqueeze(1).unsqueeze(1)
# GAMEPAD_TACTILE_MAX = torch.Tensor([15.1780, 22.9789, 20.2003]).unsqueeze(1).unsqueeze(1)

# Task based stats
TACTILE_IMAGE_STATS = {
    'gamepad': {
        'mean': [-12.8636,  -1.1549,   1.3344],
        'std': [66.7323, 43.5684, 49.4381],
        'min': [-14.7925, -22.9259, -20.2543],
        'max': [15.1780, 22.9789, 20.2003] 
    },
    'play_data': {
        'mean': [-3.4821,  0.8840, 11.0220],
        'std': [96.1595, 67.3064, 72.4123], 
        'min': [ -5.9464, -10.1432,  -9.6585],
        'max': [ 5.9450, 10.1289,  9.4191]
    }
}

# Alexnet means and stds
TACTILE_IMAGE_MEANS = [0.485, 0.456, 0.406]
TACTILE_IMAGE_STDS = [0.229, 0.224, 0.225]

TACTILE_PLAY_DATA_CLAMP_MIN = -1000
TACTILE_PLAY_DATA_CLAMP_MAX = 1000

# Actual after normalization image means and stds 
# TACTILE_IMAGE_MEANS = [0.4872, 0.5479, 0.3916]
# TACTILE_IMAGE_STDS = [0.0046, 0.0022, 0.0046]

ALLEGRO_FINGERTIP_MEANS = [
    0.0745204,
    0.02438364,
    0.10162595
]

ALLEGRO_FINGERTIP_STDS = [
    0.02147618,
    0.02644624,
    0.02051845
]

VISION_IMAGE_MEANS = [
    0.4191,
    0.4445,
    0.4409
]

VISION_IMAGE_STDS = [
    0.2108,
    0.1882,
    0.1835
]
import hydra 
from torch.nn.parallel import DistributedDataParallel as DDP

from tactile_learning.models.custom import TactileJointLinear
from tactile_learning.models.ssl_wrappers.byol import BYOL
from tactile_learning.models.learners.byol import TactileImageBYOL
from tactile_learning.utils.augmentations import get_tactile_augmentations
from tactile_learning.utils.constants import *

def init_learner(cfg, device, rank):
    if cfg.learner_type == 'bc':
        return init_bc(cfg, device, rank)
    if cfg.learner_type == 'byol':
        return init_byol(cfg, device, rank)
    return None

def init_byol(cfg, device, rank):
    # Start the encoder
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)

    augment_fn = get_tactile_augmentations(
        img_means = TACTILE_IMAGE_MEANS,
        img_stds = TACTILE_IMAGE_STDS
    )
    # Initialize the byol wrapper
    byol = BYOL(
        net = encoder,
        image_size = cfg.tactile_image_size,
        augment_fn = augment_fn
    ).to(device)
    encoder = DDP(encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    
    # Initialize the optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = byol.parameters())
    
    # Initialize the agent
    learner = TactileImageBYOL(
        byol = byol,
        optimizer = optimizer
    )

    learner.to(device)

    return learner

def init_bc(cfg, device, rank): 
    model = TactileJointLinear(input_dim=cfg.tactile_info_dim + cfg.joint_pos_dim,
                               output_dim=cfg.joint_pos_dim,
                               hidden_dim=cfg.hidden_dim).to(device) # Velocity for each joint
    
    model = DDP(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    # Initialize the optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = model.parameters())
    
    # Initialize the total agent
    learner = hydra.utils.instantiate(cfg.agent, # TODO: This should be changed
                                    model = model,
                                    optimizer = optimizer)
                                    
    learner.to(device)

    return learner

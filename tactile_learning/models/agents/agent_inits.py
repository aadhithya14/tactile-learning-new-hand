import hydra 
from torch.nn.parallel import DistributedDataParallel as DDP

from tactile_learning.models.custom import TactileJointLinear
# import tactile_learning.models.agents.behavior_cloning import TactileJointBC

def init_agent(cfg, device, rank):
    if cfg.agent_type == 'bc':
        return init_bc(cfg, device, rank)
    return None

def init_bc(cfg, device, rank): 
    model = TactileJointLinear(input_dim=cfg.tactile_info_dim + cfg.joint_pos_dim,
                               output_dim=cfg.joint_pos_dim,
                               hidden_dim=cfg.hidden_dim).to(device) # Velocity for each joint
    
    model = DDP(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    # Initialize the optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = model.parameters())
    
    # Initialize the total agent
    agent = hydra.utils.instantiate(cfg.agent,
                                    model = model,
                                    optimizer = optimizer)
                                    
    agent.to(device)

    return agent

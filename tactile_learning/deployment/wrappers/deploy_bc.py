import numpy as np 
import os
import torch 

from omegaconf import DictConfig, OmegaConf
from tactile_learning.deployment.load_models import load_model


class DeployBC:
    def __init__(self, out_dir):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"

        torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
        torch.cuda.set_device(0)

        self.device = torch.device('cuda:0')
        self.cfg = OmegaConf.load(os.path.join(out_dir, '.hydra/config.yaml'))
        model_path = os.path.join(out_dir, 'models/bc_model_best.pt')

        self.model = load_model(self.cfg, self.device, model_path)
        self.model.eval() 

    def get_action(self, tactile_info, joint_state):
        flattened_tactile = torch.reshape(tactile_info, (1,-1))
        joint_state = torch.reshape(joint_state, (1,-1))
        all_info = torch.cat((flattened_tactile, joint_state), -1)

        pred_action = self.model(all_info).squeeze()

        return pred_action
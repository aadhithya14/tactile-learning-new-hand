# Main training script - trains distributedly accordi

import glob
import os
import hydra
import logging
import wandb

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm 


# Custom imports 
from tactile_learning.utils.logger import Logger
from tactile_learning.datasets.dataloaders import get_dataloaders
from tactile_learning.models.learners.learner_inits import init_learner
from tactile_learning.utils.parsers import *
from tactile_learning.datasets.preprocess import *

class Workspace:
    def __init__(self, cfg : DictConfig) -> None:
        print(f'Workspace config: {OmegaConf.to_yaml(cfg)}')

        # Initialize hydra
        self.hydra_dir = HydraConfig.get().run.dir

        # Create the checkpoint directory - it will be inside the hydra directory
        cfg.checkpoint_dir = os.path.join(self.hydra_dir, 'models')
        os.makedirs(cfg.checkpoint_dir, exist_ok=True) # Doesn't give an error if dir exists when exist_ok is set to True 
        
        # Set the world size according to the number of gpus
        cfg.num_gpus = torch.cuda.device_count()
        print(f"cfg.num_gpus: {cfg.num_gpus}")
        print()
        cfg.world_size = cfg.world_size * cfg.num_gpus

        # Set device and config
        self.cfg = cfg

    def train(self, rank) -> None:
        # Create default process group
        dist.init_process_group("gloo", rank=rank, world_size=self.cfg.world_size)
        dist.barrier() # Wait for all of the processes to start
        
        # Set the device
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        print(f"INSIDE train: rank: {rank} - device: {device}")

        # It looks at the datatype type and returns the train and test loader accordingly
        train_loader, test_loader, _ = get_dataloaders(self.cfg)

        # Initialize the learner - looks at the type of the agent to be initialized first
        print('device: {}, rank: {}'.format(device, rank))
        learner = init_learner(self.cfg, device, rank)

        best_loss = torch.inf 

        # Logging
        if rank == 0:
            pbar = tqdm(total=self.cfg.train_epochs)
            # Initialize logger (wandb)
            if self.cfg.logger:
                wandb_exp_name = '-'.join(self.hydra_dir.split('/')[-2:])
                self.logger = Logger(self.cfg, wandb_exp_name, out_dir=self.hydra_dir)

        # Start the training
        for epoch in range(self.cfg.train_epochs):
            # Distributed settings
            if self.cfg.distributed:
                train_loader.sampler.set_epoch(epoch)
                dist.barrier()

            # Train the models for one epoch
            train_loss = learner.train_epoch(train_loader)

            if self.cfg.distributed:
                dist.barrier()

            if rank == 0: # Will only print after everything is finished
                pbar.set_description(f'Epoch {epoch}, Train loss: {train_loss:.5f}, Best loss: {best_loss:.5f}')
                pbar.update(1) # Update for each batch

            # Logging
            if self.cfg.logger and rank == 0 and epoch % self.cfg.log_frequency == 0:
                self.logger.log({'epoch': epoch,
                                 'train loss': train_loss})

            # Testing and saving the model
            if epoch % self.cfg.save_frequency == 0 and rank == 0: # NOTE: Not sure why this is a problem but this could be the fix
                # Test for one epoch
                if not self.cfg.self_supervised:
                    test_loss = learner.test_epoch(test_loader)
                else:
                    test_loss = train_loss # In BYOL (for ex) test loss is not important

                # Get the best loss
                if test_loss < best_loss:
                    best_loss = test_loss
                    learner.save(self.cfg.checkpoint_dir)

                # Logging
                if rank == 0:
                    pbar.set_description(f'Epoch {epoch}, Test loss: {test_loss:.5f}')
                    if self.cfg.logger:
                        self.logger.log({'epoch': epoch,
                                        'test loss': test_loss})
                        self.logger.log({'epoch': epoch,
                                        'best loss': best_loss})

        if rank == 0: 
            pbar.close()

@hydra.main(version_base=None,config_path='tactile_learning/configs', config_name = 'train')
def main(cfg : DictConfig) -> None:
    # TODO: check this and make it work when it's not distributed as well
    # preprocess_opt = choose_preprocess()
    print('CFG.PREPROCESS: {}'.format(cfg.preprocess))
    assert cfg.distributed is True, "Use script only to train distributed"
    workspace = Workspace(cfg)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"

    if cfg.preprocess:
        roots = glob.glob(f'{cfg.data_dir}/demonstration_*') 
        roots = sorted(roots)
        for demo_id, root in enumerate(roots):
            # dump_video_to_images(root)
            dump_fingertips(root)
            if cfg.learner_type == 'byol': # If it is byol then there are more tactile images
                dump_data_indices(demo_id, root, is_byol=True)
            else:
                dump_data_indices(demo_id, root, is_byol=False)
            print('----------------')
    
    print("Distributed training enabled. Spawning {} processes.".format(workspace.cfg.world_size))
    mp.spawn(workspace.train, nprocs=workspace.cfg.world_size)
    
if __name__ == '__main__':
    main()
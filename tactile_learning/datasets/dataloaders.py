import os
import torch
import torch.utils.data as data 
from omegaconf import DictConfig, OmegaConf

from tactile_learning.datasets.tactile import TactileDataset
from tactile_learning.datasets.preprocess import dump_video_to_images

# Script to return dataloaders
def get_dataloaders(cfg : DictConfig):
    # Load dataset - splitting will be done with random splitter
    
    dataset = TactileDataset(cfg.data_dir)

    train_dset_size = int(len(dataset) * cfg.train_dset_split)
    test_dset_size = len(dataset) - train_dset_size

    # Random split the train and validation datasets
    train_dset, test_dset = data.random_split(dataset, 
                                             [train_dset_size, test_dset_size],
                                             generator=torch.Generator().manual_seed(cfg.seed))
    train_sampler = data.DistributedSampler(train_dset, drop_last=True, shuffle=True) if cfg.distributed else None
    test_sampler = data.DistributedSampler(test_dset, drop_last=True, shuffle=False) if cfg.distributed else None # val will not be shuffled

    train_loader = data.DataLoader(train_dset, batch_size=cfg.batch_size, shuffle=train_sampler is None,
                                    num_workers=cfg.num_workers, sampler=train_sampler)
    test_loader = data.DataLoader(test_dset, batch_size=cfg.batch_size, shuffle=test_sampler is None,
                                    num_workers=cfg.num_workers, sampler=test_sampler)

    return train_loader, test_loader, dataset

if __name__ == "__main__":
    # Start the multiprocessing to load the saved models properly
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"

    torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
    torch.cuda.set_device(0)
    

    cfg = OmegaConf.load('/home/irmak/Workspace/tactile-learning/tactile_learning/configs/train.yaml')
    # cfg = OmegaConf.load('/home/irmak/Workspace/DAWGE/contrastive_learning/configs/train.yaml')
    dset = TactileDataset(
        data_path = cfg.data_dir
    )

    train_loader, test_loader, _ = get_dataloaders(cfg)
    batch = next(iter(train_loader))
    print('batch[0].shape: {}, batch[1].shape: {}, batch[2].shape: {}, batch[3].shape: {}'.format(
        batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape
    ))

    # action_min, action_max, corner_min, corner_max = dset.calculate_mins_maxs()
    # print('action: [min: {}, max: {}], corners: [min: {}, max: {}]'.format(
    #     action_min, action_max, corner_min, corner_max
    # ))



    # batch = next(iter(test_loader))
    # pos, next_pos, action = [b for b in batch]
    # print('pos: {}'.format(pos))
    # print(dset.denormalize_corner(pos[0].detach().numpy()))
    # print(dset.denormalize_action(action))
    
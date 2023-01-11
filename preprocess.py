import glob
import hydra 
from omegaconf import DictConfig

from tactile_learning.datasets.preprocess import *

@hydra.main(version_base=None, config_path='tactile_learning/configs', config_name='preprocess')
def main(cfg : DictConfig) -> None:
    # data_path = '/home/irmak/Workspace/Holo-Bot/extracted_data/box_handle_lifting/box_location_changing'
    roots = glob.glob(f'{cfg.data_path}/demonstration_*') # TODO: change this in the future
    roots = sorted(roots)

    for demo_id, root in enumerate(roots):
        if cfg.vision_byol:
            dump_video_to_images(root, view_num=cfg.view_num)
        dump_fingertips(root=root)
        dump_data_indices(demo_id=demo_id, root=root, is_byol_tactile=cfg.tactile_byol, is_byol_image=cfg.vision_byol)
        print('-----')    

if __name__ == '__main__':
    main()
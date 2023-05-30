import glob
import hydra 
from omegaconf import DictConfig

from tactile_learning.datasets import *

@hydra.main(version_base=None, config_path='tactile_learning/configs', config_name='preprocess')
def main(cfg : DictConfig) -> None:
    if cfg.process_single_demo:
        roots = [cfg.data_path]
    else:
        roots = glob.glob(f'{cfg.data_path}/demonstration_*') # TODO: change this in the future
        roots = sorted(roots)
    
    for demo_id, root in enumerate(roots):
        if cfg.dump_fingertips:
            dump_fingertips(root=root)
        if cfg.dump_data_indices:
            dump_data_indices(
                demo_id = demo_id, 
                root = root, 
                is_byol_tactile = cfg.tactile_byol, 
                is_byol_image = cfg.vision_byol, 
                threshold_step_size = cfg.threshold_step_size,
                cam_view_num = cfg.view_num,
                subsample_separately = cfg.subsample_separately,
                kinova_threshold = cfg.kinova_threshold_step_size,
                allegro_threshold = cfg.allegro_threshold_step_size,
                shorten_demos = cfg.shorten_demo.apply,
                shortening_times = cfg.shorten_demo.times,
            )
        if cfg.vision_byol:
            dump_video_to_images(root, view_num=cfg.view_num, dump_all=True) # If dump_all == False then it will use the desired images only
        elif cfg.dump_images:
            dump_video_to_images(root, view_num=cfg.view_num, dump_all=False)
        

        print('-----')    

    if cfg.repr_preprocessor.apply:
        preprocessor = RepresentationPreprocessor(
            data_path = cfg.data_path,
            tactile_out_dir = cfg.repr_preprocessor.tactile_out_dir, 
            image_out_dir = cfg.repr_preprocessor.image_out_dir,
            view_num = cfg.view_num,
            demos_to_use = cfg.repr_preprocessor.demos_to_use,
            representation_types=cfg.repr_preprocessor.representation_types
        )
        preprocessor.get_all_representations()
        print(f'Dumping - all_representations.shape: {preprocessor.all_representations.shape}')
        preprocessor.dump_all_representations()

if __name__ == '__main__':
    main()
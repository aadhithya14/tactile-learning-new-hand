import glob
import numpy as np
import os
import torch
import torchvision.transforms as T 

from torch.utils import data
from torchvision.datasets.folder import default_loader as loader 

from tactile_dexterity.tactile_data import TactileImage
from tactile_dexterity.utils import VISION_IMAGE_MEANS, VISION_IMAGE_STDS, load_data, crop_transform

# Sequential Dataset
class BetDataset(data.Dataset):
    # Dataset for bet training
    def __init__(
        self,
        seq_length,
        tactile_encoder,
        image_encoder,
        data_path,
        tactile_img_size,
        vision_view_num
    ):
        
        super().__init__()
        self.roots = glob.glob(f'{data_path}/demonstration_*')
        self.roots = sorted(self.roots)
        self.data = load_data(self.roots, demos_to_use=[])
        self.tactile_information_type = 'whole_hand'
        self.vision_view_num = vision_view_num
        self.seq_length = seq_length
        self.tactile_encoder = tactile_encoder 
        self.image_encoder = image_encoder

        tactile_cfg, tactile_encoder, _ = self._init_encoder_info(device, tactile_out_dir, 'tactile')
        self.tactile_img = TactileImage(
            tactile_image_size = tactile_cfg.tactile_image_size, 
            shuffle_type = tactile_shuffle_type
        )\]
        self.tactile_repr = TactileRepresentation(
            encoder_out_dim = tactile_cfg.encoder.out_dim,
            tactile_encoder = tactile_encoder,
            tactile_image = self.tactile_img,
            representation_type = tactile_repr_type
        )

        self.vision_transform = T.Compose([
            T.Resize((480,640)),
            T.Lambda(crop_transform),
            T.ToTensor(),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS),
        ])

        # Set the indices for one sensor
        if tactile_information_type == 'single_sensor':
            self._preprocess_tactile_indices()
    
        self.tactile_img = TactileImage(
            tactile_image_size = tactile_img_size,
            shuffle_type = None
        )

    def _preprocess_tactile_indices(self):
        self.tactile_mapper = np.zeros(len(self.data['tactile']['indices'])*15).astype(int)
        for data_id in range(len(self.data['tactile']['indices'])):
            for sensor_id in range(15):
                self.tactile_mapper[data_id*15+sensor_id] = data_id # Assign each finger to an index basically

    def _get_sensor_id(self, index):
        return index % 15
    
    def __len__(self):
        if self.tactile_information_type == 'single_sensor':
            return len(self.tactile_mapper)
        else: 
            return len(self.data['tactile']['indices'])
        
    def _get_proper_tactile_value(self, index):
        if self.tactile_information_type == 'single_sensor':
            data_id = self.tactile_mapper[index]
            demo_id, tactile_id = self.data['tactile']['indices'][data_id]
            sensor_id = self._get_sensor_id(index)
            tactile_value = self.data['tactile']['values'][demo_id][tactile_id][sensor_id]
            
            return tactile_value
        
        else:
            demo_id, tactile_id = self.data['tactile']['indices'][index]
            tactile_values = self.data['tactile']['values'][demo_id][tactile_id]
            
            return tactile_values

    def _get_image_repr(self, index):
        demo_id, image_id = self.data['image']['indices'][index]
        image_root = self.roots[demo_id]
        image_path = os.path.join(image_root, 'cam_{}_rgb_images/frame_{}.png'.format(self.vision_view_num, str(image_id).zfill(5)))
        img = self.vision_transform(loader(image_path))

        vision_img = torch.FloatTensor(img)

        return self.image_encoder(vision_img.unsqueeze(0))

    def _get_tactile_image_repr(self, tactile_values):
        tactile_img = self.tactile_img.get(
            type = self.tactile_information_type,
            tactile_values = tactile_values
        )

        # Get the representation

        return self.tactile_img.get(
            type = self.tactile_information_type,
            tactile_values = tactile_values
        )

    # Gets the kinova states and the commanded joint states for allegro
    def _get_action(self, index):
        demo_id, allegro_action_id = self.data['allegro_actions']['indices'][index]
        allegro_action = self.data['allegro_actions']['values'][demo_id][allegro_action_id]

        _, kinova_id = self.data['kinova']['indices'][index]
        kinova_action = self.data['kinova']['values'][demo_id][kinova_id]

        total_action = np.concatenate([allegro_action, kinova_action], axis=-1)
        return torch.FloatTensor(total_action) # These values are already quite small so we'll not normalize them

    def __getitem__(self, index):

        # Traverse through for each sequence
        for seq_id in range(self.seq_length):

            tactile_value = self._get_proper_tactile_value(index+seq_id)

            if seq_id == 0:
                tactile_images = self._get_tactile_image(tactile_value).unsqueeze(0)
                vision_images = self._get_image(index+seq_id).unsqueeze(0)
                actions = self._get_action(index+seq_id).unsqueeze(0)
            
            else:
                tactile_images = torch.concat([
                    tactile_images,
                    self._get_tactile_image(tactile_value).unsqueeze(0)
                ], dim=0)
                vision_images = torch.concat([
                    vision_images,
                    self._get_image(index+seq_id).unsqueeze(0)
                ], dim=0)
                actions = torch.concat([
                    actions,
                    self._get_action(index+seq_id).unsqueeze(0)
                ], dim=0)
                
        # Concatenate tactile and vision images
        obs = torch.concat([tactile_images, vision_images], dim=0)
        return obs, actions
    

if __name__ == '__main__':
    # Test the dataset 
    dset = SequentialDataset(
        seq_length=10,
        data_path = '/home/irmak/Workspace/Holo-Bot/extracted_data/cup_picking/after_rss',
        tactile_information_type = 'whole_hand',
        tactile_img_size=480,
        vision_view_num=1
    ) 
    print(len(dset))
    dataloader = data.DataLoader(dset, 
                                batch_size  = 16, 
                                shuffle     = True, 
                                num_workers = 8,
                                pin_memory  = True)

    batch = next(iter(dataloader))
    print('batch[0].shape: {}, batch[1].shape: {}'.format(
        batch[0].shape, batch[1].shape # it should be 16 + 7 (for each joint)
    ))

    
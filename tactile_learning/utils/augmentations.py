import torch.nn as nn
import torchvision.transforms as T 

# Method for tactile augmentations
def get_tactile_augmentations(img_means, img_stds, img_size=(8,8)):
    tactile_aug = T.Compose([
        T.RandomApply(
            nn.ModuleList([T.RandomResizedCrop(img_size, scale=(.8, 1))]),
            p = 0.5
        ), 
        T.RandomApply(
            nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), 
            p = 0.5
        ),
        T.Normalize(
            mean = img_means, # NOTE: Wasn't this wrong?
            std = img_stds
        )
    ])
    return tactile_aug

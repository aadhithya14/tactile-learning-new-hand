import torch.nn as nn
import torchvision.transforms as T 
from torchvision.transforms.functional import crop

# Method for tactile augmentations
def get_tactile_augmentations(img_means, img_stds, img_size):
    tactile_aug = T.Compose([
        T.RandomApply(
            nn.ModuleList([T.RandomResizedCrop(img_size, scale=(.9, 1))]),
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

def get_vision_augmentations(img_means, img_stds):
    color_aug = T.Compose([
        T.RandomApply(
            nn.ModuleList([T.ColorJitter(.8,.8,.8,.2)]), 
            p = 0.2
        ),
        T.RandomGrayscale(p=0.2), 
        T.RandomApply(
            nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), 
            p = 0.2
        ),
        T.Normalize(
            mean = img_means, # NOTE: Wasn't this wrong?
            std =  img_stds
        )
    ])

    return color_aug 

def crop_transform(image):
    return crop(image, 0,90,480,480)
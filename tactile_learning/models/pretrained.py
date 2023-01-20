from torchvision import models
import torch.nn as nn
import torch

# Script to return all pretrained models in torchvision.models module
def resnet18(pretrained : bool):
    encoder = models.__dict__['resnet18'](pretrained = True)
    encoder.fc = nn.Identity()

    return encoder

def resnet34(pretrained : bool):
    encoder = models.__dict__['resnet34'](pretrained = True)
    encoder.fc = nn.Identity()

    return encoder


def alexnet(pretrained, out_dim, remove_last_layer=False):
    encoder = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=pretrained)

    if remove_last_layer:
        # Remove and recreate the last layer of alexnet - should be 
        encoder.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(9216, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, out_dim, bias=True)
        )

    return encoder
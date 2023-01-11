from torchvision import models
import torch.nn as nn

# Script to return all pretrained models in torchvision.models module
def resnet18(pretrained : bool):
    if pretrained:
        encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        encoder = models.resnet18()
    encoder.fc = nn.Identity()

    return encoder

def resnet34(pretrained : bool):
    if pretrained:
        encoder = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    else:
        encoder = models.resnet34()
    encoder.fc = nn.Identity()
    return encoder
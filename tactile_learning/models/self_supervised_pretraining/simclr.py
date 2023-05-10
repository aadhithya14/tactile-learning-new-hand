import torch
import torch.nn as nn

from tactile_learning.utils.losses import nt_xent_loss

class SimCLR(nn.Module):
    def __init__(self,
        encoder,
        projector,
        augment_fn,
        sec_augment_fn,
        temperature,
        device
    ):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.projector = projector
        self.augment_fn = augment_fn
        self.sec_augment_fn = sec_augment_fn
        self.temperature = temperature

        # Send a mock image tensor to instantiate singleton parameters
        self.to(device)
        self.forward(
            image = torch.randn(2, 3, 224, 224, device=device)
        )

    def parameters(self):
        return list(list(self.encoder.parameters()) + list(self.projector.parameters()))

    def to(self, device):
        self.encoder.to(device)
        self.projector.to(device)

    def train(self):
        self.encoder.train()
        self.projector.train()

    def eval(self):
        self.encoder.eval()
        self.projector.eval()

    def get_image_representation(self, image, is_second_img = False):
        if is_second_img:
            augmented_image = self.sec_augment_fn(image)
        else:
            augmented_image = self.augment_fn(image)
        
        representation = self.encoder(augmented_image)
        return representation

    def forward(self, image):
        first_projection = self.get_image_representation(image, is_second_img = False)
        second_projection = self.get_image_representation(image, is_second_img = True)

        first_expanded_projection = self.projector(first_projection)
        second_expanded_projection = self.projector(second_projection)

        loss = nt_xent_loss(
            first_expanded_projection,
            second_expanded_projection,
            self.temperature
        )
        return loss 

    def get_encoder_weights(self):
        return self.encoder.state_dict()
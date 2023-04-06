# Model for the Gaussian Mixture Model implementation
import torch 
import torch.nn as nn

from .utils import create_fc

class GMMLayer(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        num_gaussians,
        hidden_dims
    ):
        super().__init__()
        self.A = action_dim
        self.G = num_gaussians
        self.model = create_fc(
            input_dim = obs_dim,
            output_dim = 2 * num_gaussians * action_dim + num_gaussians, # We will predict 3 different numbers to generate gaussian clusters for each action space
            hidden_dims = hidden_dims
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # Input will be the observation representation
        output_repr = self.model(x) # This will have the dimension (B,G(1 + 2A)) G for logits, AG for mu AG for sigma 
        logits = output_repr[:,:self.G]
        mu = output_repr[:,self.G:self.G*(1+self.A)].reshape((-1, self.G, self.A))
        
        log_sigma = output_repr[:,self.G*(1+self.A):self.G*(1+2*self.A)].reshape((-1, self.G, self.A))
        log_sigma = torch.clamp(
            log_sigma, -5, 2
        )
        sigma = torch.exp(log_sigma)

        return logits, mu, sigma


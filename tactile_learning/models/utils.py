import numpy as np
import torch
import torch.nn as nn

def create_fc(input_dim, output_dim, hidden_dims, use_batchnorm=False, dropout=None, is_moco=False):
    if hidden_dims is None:
        return nn.Sequential(*[nn.Linear(input_dim, output_dim)])

    layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]

    if use_batchnorm:
        layers.append(nn.BatchNorm1d(hidden_dims[0]))

    if dropout is not None:
        layers.append(nn.Dropout(p = dropout))

    for idx in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
        layers.append(nn.ReLU())

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dims[idx + 1]))

        if dropout is not None:
            layers.append(nn.Dropout(p = dropout))

    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    if is_moco:
        layers.append(nn.BatchNorm1d(hidden_dims[-1], affine=False))
    return nn.Sequential(*layers)

# Ref: https://github.com/denisyarats/drq/blob/master/utils.py
def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

# Ref: https://github.com/denisyarats/drq/blob/master/utils.py
def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)
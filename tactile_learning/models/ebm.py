import dataclasses
import enum
from functools import partial
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivationType(enum.Enum):
    RELU = nn.ReLU
    SELU = nn.SiLU

# Taken from https://github.com/kevinzakka/ibc/ implementation
# Script to see the deployment
@dataclasses.dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    hidden_depth: int
    dropout_prob: Optional[float] = None
    activation_fn: ActivationType = ActivationType.RELU

class MLP(nn.Module):
    """A feedforward multi-layer perceptron."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()

        dropout_layer: Callable
        dropout_layer: Callable
        if config.dropout_prob is not None:
            dropout_layer = partial(nn.Dropout, p=config.dropout_prob)
        else:
            dropout_layer = nn.Identity

        layers: Sequence[nn.Module]
        if config.hidden_depth == 0:
            layers = [nn.Linear(config.input_dim, config.output_dim)]
        else:
            layers = [
                nn.Linear(config.input_dim, config.hidden_dim),
                config.activation_fn.value(),
                dropout_layer(),
            ]
            for _ in range(config.hidden_depth - 1):
                layers += [
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    config.activation_fn.value(),
                    dropout_layer(),
                ]
            layers += [nn.Linear(config.hidden_dim, config.output_dim)]
        layers = [layer for layer in layers if not isinstance(layer, nn.Identity)]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class EBMMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        hidden_depth,
        dropout_prob=0
    ) -> None:
        super().__init__()
        mlp_config = MLPConfig(
            input_dim = input_dim, 
            hidden_dim = hidden_dim,
            output_dim = output_dim,
            hidden_depth = hidden_depth,
            dropout_prob=dropout_prob
        )
        self.mlp = MLP(mlp_config)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([x.unsqueeze(1).expand(-1, y.size(1), -1), y], dim=-1)
        B, N, D = fused.size()
        fused = fused.reshape(B * N, D).float()
        out = self.mlp(fused)
        return out.view(B,N)
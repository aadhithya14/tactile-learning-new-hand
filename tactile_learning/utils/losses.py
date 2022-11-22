import torch
import torch.nn.functional as F

def l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(x, y)

def mse(x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, y)
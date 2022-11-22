import torch 
import torch.nn as nn

class PrintSize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # print("mean: {}".format(x.mean()))
        print(x.shape)
        return x

# Linear module to get the flattened tactile info and joint positions only
# and return action - bc model 
class TactileJointLinear(nn.Module):
    def __init__(
        self,
        input_dim:int=736, # 16*15*3 + 16
        output_dim:int=16,
        hidden_dim:int=64
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*3),
            nn.ReLU(),
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid() # Should be mapped bw -1,1
        )

    def forward(self, x):
        action = self.model(x)
        return action
    
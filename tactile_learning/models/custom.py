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
            # nn.Sigmoid() # Should be mapped bw -1,1
        ) # TODO: Check the activation functions! 

    def forward(self, x):
        action = self.model(x.float())
        return action

class TactileLinearEncoder(nn.Module):
    def __init__(
        self,
        input_dim = 48,
        hidden_dim = 128,
        output_dim = 64
    ):
        super().__init__() 
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # Flatten the image
        x = torch.flatten(x,1)
        x = self.model(x)
        return x

class TactileSingleSensorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_dim # Final dimension of the representation
    ):
        super().__init__()
        self.out_dim = out_dim
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=2),
            nn.ReLU(),
            # PrintSize(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2),
            nn.ReLU(),
            # PrintSize(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2),
            nn.ReLU(),
            # PrintSize()
        )
        self.linear = nn.Linear(in_features=128, out_features=out_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = self.linear(x)
        return self.relu(x)

class TactileImageEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_dim # Final dimension of the representation
    ):
        super().__init__()
        self.out_dim = out_dim
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=2),
            nn.ReLU(),
            # PrintSize(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2),
            nn.ReLU(),
            # PrintSize(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2),
            nn.ReLU(),
            # PrintSize()
        )
        self.linear = nn.Linear(in_features=16*5*5, out_features=out_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = self.linear(x)
        return self.relu(x)

class TactileLargeImageEncoder(nn.Module): # Encoder for the whole tactile image
    def __init__(
        self,
        in_channels,
        out_dim # Final dimension of the representation
    ):
        super().__init__()
        self.out_dim = out_dim
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=2),
            nn.ReLU(),
            # PrintSize(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4),
            nn.ReLU(),
            # PrintSize(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2),
            nn.ReLU(),
            # PrintSize(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2),
            nn.ReLU(),
            # PrintSize()
        )
        self.linear = nn.Linear(in_features=16*10*10, out_features=out_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = self.linear(x)
        return self.relu(x)

class TactileStackedImageEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_dim # Final dimension of the representation
    ):
        super().__init__()
        self.out_dim = out_dim
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=2),
            nn.ReLU(),
            PrintSize(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2),
            nn.ReLU(),
            PrintSize(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2),
            nn.ReLU(),
            PrintSize()
        )
        self.linear = nn.Linear(in_features=16*5*5, out_features=out_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = self.linear(x)
        return self.relu(x)
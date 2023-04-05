import torch 
import torch.nn.functional as F

def get_tactile_image(tactile_values):
    # tactile_values: (N,16,3) - N is the included number of sensors
    # TODO: This method should be robust to all possible shapes and etc
    tactile_image = torch.FloatTensor(tactile_values)
    if len(tactile_values) <= 2:
        tactile_image = tactile_image.reshape((len(tactile_values),4,4,-1))
        tactile_image = torch.concat((tactile_image[0], tactile_image[1]), dim=1)
    elif len(tactile_values) == 15:
        # pad it to 16
        tactile_image = F.pad(tactile_image, (0,0,0,0,1,0), 'constant', 0)
        # reshape it to 4x4
        tactile_image = tactile_image.view(16,4,4,3)

        # concat for it have its proper shape
        tactile_image = torch.concat([
            torch.concat([tactile_image[i*4+j] for j in range(4)], dim=0)
            for i in range(4)
        ], dim=1)

    tactile_image = torch.permute(tactile_image, (2,0,1))

    return tactile_image
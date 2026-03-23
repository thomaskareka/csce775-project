import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register_model

@register_model("atari_cnn")
class AtariCNN(nn.Module):
    def __init__(self, input_shape, num_actions, action_type):
        super().__init__()
        c, h, w = input_shape

        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.conv_output_size = self._get_conv_output(input_shape)
        self.fc1 = nn.Linear(self.conv_output_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    # dynamic output calculation, rather than assuming 84x84
    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            output = x.view(1, -1).size(1)
            print(output)
            return output
    
    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)
        

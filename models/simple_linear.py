import torch.nn as nn
from models import register_model

@register_model("simple_linear")
class SimpleLinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_type, num_actions):
        super().__init__()

        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, num_actions)

        self.act = nn.ReLU()
    
    def forward(self, x):
        # flatten image observations
        x = x.view(x.size(0), -1)

        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
        x = self.lin3(x)

        return x
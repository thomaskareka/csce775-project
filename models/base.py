import torch.nn as nn
from models import register_model

@register_model("base")
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError
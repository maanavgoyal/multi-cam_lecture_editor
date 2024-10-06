import torch
from torch import nn
from torchvision.models.video import swin3d_t

class VideoSwinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = swin3d_t(pretrained=True)
        # Remove the classification head
        self.model.head = nn.Identity()

    def forward(self, x):
        return self.model(x)
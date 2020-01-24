import torch.nn as nn

from .frontend import VisualFrontEnd
from .backends import ConvolutionalBackend


class PretrainNet(nn.Module):
    def __init__(self, resnet, nh):
        super().__init__()
        self.frontend = VisualFrontEnd(out_channels=nh, resnet=resnet)
        self.backend = ConvolutionalBackend(nh, 500)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

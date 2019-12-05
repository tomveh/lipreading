import torch.nn as nn

from .frontend import VisualFrontEnd
from .backends import ConvolutionalBackend, LSTMBackend


class LipreadingResNet(nn.Module):
    def __init__(self, resnet, backend, nh=256):
        super().__init__()
        self.frontend = VisualFrontEnd(out_channels=nh, resnet=resnet)

        if backend == 'conv':
            self.backend = ConvolutionalBackend(nh, 500)
        elif backend == 'lstm':
            self.backend = LSTMBackend(nh, 500)
        else:
            raise RuntimeError(f'Unknown backend type: {backend}')

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

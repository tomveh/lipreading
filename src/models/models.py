import torch.nn as nn

from .frontend import VisualFrontEnd
from .backends import ConvolutionalBackend, TransformerBackend


class PretrainNet(nn.Module):
    def __init__(self, resnet, nh):
        super().__init__()
        self.frontend = VisualFrontEnd(out_channels=nh, resnet=resnet)
        self.backend = ConvolutionalBackend(nh, 500)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab, resnet, nh):
        super().__init__()
        self.vocab = vocab
        self.frontend = VisualFrontEnd(out_channels=nh, resnet=resnet)
        self.backend = TransformerBackend(vocab, nh)

    def forward(self, x, y):
        x = self.frontend(x)
        x = self.backend(x, y)
        return x

    def inference(self, x):
        x = self.frontend(x)
        x = self.backend.inference(x)

        return x

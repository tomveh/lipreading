import torch.nn as nn

from .backends import ConvolutionalBackend, TransformerBackend
from .crazy_stuff import Model
from .frontend import SpatioTemporalFrontend, VisualFrontend


class PretrainNet(nn.Module):
    def __init__(self, resnet, nh):
        super().__init__()
        if resnet == 'resnet18':
            self.frontend = VisualFrontend()
        elif resnet == '3d':
            self.frontend = Model()
        self.backend = ConvolutionalBackend(nh, 500)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab, resnet):
        super().__init__()
        self.vocab = vocab
        self.frontend = VisualFrontend()
        self.backend = TransformerBackend(vocab)

    def forward(self, x, y=None):
        x = self.frontend(x.unsqueeze(1))
        x = self.backend(x, y)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, frontend='resnet', num_layers=6, nhead=8, d_model=512):
        super().__init__()

        assert frontend in ['resnet', '3d', '2plus1']

        if frontend == 'resnet':
            self.frontend = VisualFrontend()
        elif frontend == '3d':
            self.frontend = SpatioTemporalFrontend()
        elif frontend == '2plus1':
            self.frontend = Model()

        if d_model != 512:
            self.linear1 = nn.Linear(512, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers)
        self.linear2 = nn.Linear(d_model, 500)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):
        x = self.frontend(x)

        if hasattr(self, 'linear1'):
            x = self.linear1(x)

        x = self.encoder(x.transpose(0, 1).contiguous())
        x = self.linear2(x).permute(
            1, 2, 0).contiguous()  # (S, N, n_out) => (N, n_out, S)
        x = self.maxpool(x).squeeze(-1)
        return x


#     # use 3d convolution frontend
# class TransformerClassifier2(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.frontend = SpatioTemporalFrontend()
#         encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
#         self.linear = nn.Linear(512, 500)
#         self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)

#     def forward(self, x):
#         x = self.frontend(x)
#         x = self.encoder(x.transpose(0, 1).contiguous())
#         x = self.linear(x).permute(
#             1, 2, 0).contiguous()  # (S, N, n_out) => (N, n_out, S)
#         x = self.maxpool(x).squeeze(-1)
# return x

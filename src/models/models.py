import torch.nn as nn

from .backends import ConvolutionalBackend, TransformerBackend
from .frontend import SpatioTemporalFrontend, ResnetFrontend


class PretrainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = ResnetFrontend()
        self.backend = ConvolutionalBackend(512, 500)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, frontend='resnet', num_layers=6, nhead=8, d_model=512):
        super().__init__()

        assert frontend in ['resnet', '3d']

        if frontend == 'resnet':
            self.frontend = ResnetFrontend()
        elif frontend == '3d':
            self.frontend = SpatioTemporalFrontend()

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


class TransformerModel(nn.Module):
    def __init__(self, vocab, resnet):
        super().__init__()
        self.vocab = vocab
        self.frontend = ResnetFrontend()
        self.backend = TransformerBackend(vocab)

    def forward(self, x, y=None):
        x = self.frontend(x.unsqueeze(1))
        x = self.backend(x, y)
        return x

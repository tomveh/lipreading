import torch
import torch.nn as nn

from .beam_search import beam_search


class ConvolutionalBackend(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels,
                      2 * input_channels,
                      kernel_size=5,
                      stride=2,
                      bias=False), nn.BatchNorm1d(2 * input_channels),
            nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(2 * input_channels,
                      4 * input_channels,
                      kernel_size=5,
                      stride=2,
                      bias=False), nn.BatchNorm1d(4 * input_channels),
            nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Sequential(
            nn.Linear(4 * input_channels, input_channels),
            nn.BatchNorm1d(input_channels), nn.ReLU(inplace=True))

        self.linear2 = nn.Linear(input_channels, output_channels)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()  # transpose channel and depth dims
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class TransformerBackend(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding = nn.Embedding(tokenizer.get_vocab_size(), 512)
        self.transformer = nn.Transformer(d_model=512)
        self.linear = nn.Linear(512, tokenizer.n_output)
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if 'Conv1d' in module.__class__.__name__:
                nn.init.kaiming_uniform_(module.weight.data)

    def forward(self, x, y=None):
        if self.training and y is not None:
            y_embedded = self.embedding(y)

            # nn.Transformer wants shapes (S, N, E)...
            src = x.transpose(0, 1).contiguous()
            # ...and (T, N, E)
            tgt = y_embedded.transpose(0, 1).contiguous()

            # not allowed to look ahead
            tgt_mask = self.transformer.generate_square_subsequent_mask(
                len(tgt)).type_as(tgt)

            src_key_padding_mask = x.sum(dim=2) == 0
            tgt_key_padding_mask = y == self.tokenizer.token_to_id('<pad>')

            out = self.transformer(src,
                                   tgt,
                                   tgt_mask=tgt_mask,
                                   src_key_padding_mask=src_key_padding_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask)

            pred = self.linear(out)

            # seq_len x batch_size x n_vocab
            # -> batch_size x n_vocab x seq_len_n_vocab
            pred = pred.permute(1, 2, 0).contiguous()

            return pred

        elif not self.training and y is None:
            with torch.no_grad():
                return beam_search(self, x, beam_width=10)

        else:
            raise RuntimeError('not sure if train or eval')

import torch.nn as nn
import torch.nn.functional as F

from .beam_search import beam_search


class ConvolutionalBackend(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels,
                               2 * input_channels,
                               kernel_size=5,
                               stride=2,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(2 * input_channels)
        self.conv2 = nn.Conv1d(2 * input_channels,
                               4 * input_channels,
                               kernel_size=5,
                               stride=2,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(4 * input_channels)
        self.linear1 = nn.Linear(4 * input_channels, input_channels)
        self.bn3 = nn.BatchNorm1d(input_channels)
        self.linear2 = nn.Linear(input_channels, output_channels)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous(
        )  # batch x depth x input_size -> batch x input_size x depth
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool1d(x, output_size=1).squeeze(-1)
        x = F.relu(self.bn3(self.linear1(x)))
        x = self.linear2(x)
        return x


class TransformerBackend(nn.Module):
    def __init__(
            self,
            vocab,
            nh,
    ):
        super().__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding(vocab.n_embed, nh)
        self.transformer = nn.Transformer(d_model=nh)
        self.linear = nn.Linear(nh, vocab.n_output)
        self._init_weights()

    def _init_weights(self):
        pass

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
            tgt_key_padding_mask = y == self.vocab.token2idx('<pad>')

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
            return beam_search(self, x, beam_width=10)

        else:
            raise RuntimeError('not sure if train or eval')

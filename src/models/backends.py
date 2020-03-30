import torch
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

    def greedy(self, x, device='cuda'):
        N = x.shape[0]

        sos = self.vocab.token2idx('<sos>')

        decoded = torch.tensor([sos] * N).reshape(N, 1).to(device)

        # shape (N, 1) indicating if sequence is finished
        done = decoded == self.vocab.token2idx('<eos>')

        src = x.transpose(0, 1)  # max_seq_len, batch, d_model
        src_key_padding_mask = x.sum(
            dim=-1) == 0  # pad input if all d_model dim values are 0

        max_len = 100

        for _ in range(max_len):
            tgt = self.embedding(decoded).transpose(0, 1).contiguous()
            # shape: (T, N)

            tgt_key_padding_mask = decoded == self.vocab.token2idx('<pad>')
            # shape: (N, T)

            # not allowed to look ahead
            tgt_mask = self.transformer.generate_square_subsequent_mask(
                len(tgt)).type_as(tgt_key_padding_mask)

            out = self.transformer(src,
                                   tgt,
                                   tgt_mask=tgt_mask,
                                   src_key_padding_mask=src_key_padding_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask)

            pred = self.linear(out)
            # shape: (seq_len x batch_size x n_vocab)

            greedy_pred = pred.argmax(dim=2)[-1].reshape(N, 1)

            # replace prediction with <pad> if eos has been predicted earlier
            # greedy_pred[done] = self.vocab.token2idx('<pad>')

            done = done | (
                greedy_pred.view(-1) == self.vocab.token2idx('<eos>'))

            decoded = torch.cat([decoded, greedy_pred], dim=1)

            if done.all():
                break

        decoded_strings = []

        # special = [
        #     self.vocab.token2idx(token)
        #     for token in ['<sos>', '<pad>', '<eos>']
        # ]

        for line in decoded:
            decoded_strings.append(''.join([
                self.vocab.idx2token(idx)
                for idx in line  # if idx not in special
            ]).replace('<sos>', '^').replace('<eos>',
                                             '$').replace('<pad>', '#'))

        return decoded_strings

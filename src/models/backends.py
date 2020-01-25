import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, vocab, d_model, frontend):
        super().__init__()
        self.vocab = vocab
        self.frontend = frontend
        self.embedding = nn.Embedding(len(vocab), d_model)
        self.transformer = nn.Transformer(d_model=d_model)
        self.linear = nn.Linear(d_model, len(vocab))

    def forward(self, x, y):

        if self.frontend:
            x = self.frontend(x)

        y_embedded = self.embedding(y)

        # nn.Transformer wants shapes (S, N, E)...
        src = x.transpose(0, 1).contiguous()
        # ...and (T, N, E)
        tgt = y_embedded.transpose(0, 1).contiguous()

        # not allowed to look ahead
        tgt_mask = self.transformer.generate_square_subsequent_mask(
            len(tgt)).cuda()

        out = self.transformer(src, tgt, tgt_mask=tgt_mask)

        pred = self.linear(out)

        # seq_len x batch_size x n_vocab -> batch_size x n_vocab x seq_len_n_vocab
        pred = pred.permute(1, 2, 0).contiguous()

        return pred

    def inference(self, x):
        with torch.no_grad():

            sos = self.vocab.token2idx('<sos>')
            pad = self.vocab.token2idx('<pad>')

            if self.frontend:
                x = self.frontend(x)

            batch_size = len(x)

            # every prediction starts with <sos>
            y = torch.tensor([sos] * batch_size).view(batch_size, 1).cuda()

            # for each sequence in batch keep track if <pad> (indicates end of sequence)
            # has been seen
            done = torch.tensor([False] * batch_size).cuda()

            seq_len = 0

            while not done.all():

                y_embedded = self.embedding(y)

                src = x.transpose(0, 1).contiguous()
                tgt = y_embedded.transpose(0, 1).contiguous()

                tgt_mask = self.transformer.generate_square_subsequent_mask(
                    len(tgt)).cuda()

                out = self.transformer(src, tgt, tgt_mask=tgt_mask)

                # transformer outputs (T, N, E)
                # where T = target seq len, N = batch size, E = embedding dim size
                #
                # let's take the last char of the predicted sequence for each batch: out[-1, :, :]
                pred = self.linear(out[-1])
                # pred is of shape (N, n_vocab)

                greedy = pred.argmax(dim=1, keepdim=True)
                # greedy is of shape (batch_size, 1) and contains
                # the index for the best token

                y = torch.cat([y, greedy], dim=1).cuda()

                seq_len += 1

                done = done | (greedy.squeeze() == pad)

                if seq_len > 100:
                    break

        return y

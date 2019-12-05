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


class LSTMBackend(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.lstm = nn.LSTM(input_size=256,
                            hidden_size=256,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)

        self.linear = nn.Linear(2 * 256, output_channels)

    def forward(self, x):

        x, _ = self.lstm(x)
        # batch x seq x 2*hidden_size

        x = self.linear(x)
        # batch x seq x 500

        # alternatively the last hidden state could be used:
        # x = x[:, -1, :]
        # but mean of all states should result in higher accuracy
        x = x.mean(dim=1)
        # batch x 500

        return x

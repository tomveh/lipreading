import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18, resnet34


class VisualFrontEnd(nn.Module):
    def __init__(self, out_channels, resnet='resnet18'):
        super().__init__()
        self.conv3d = nn.Conv3d(1,
                                64,
                                kernel_size=(5, 7, 7),
                                stride=(1, 2, 2),
                                padding=(2, 3, 3),
                                bias=False)
        self.bn1 = nn.BatchNorm3d(64)

        self.max_pool3d = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                       stride=(1, 2, 2),
                                       padding=(0, 1, 1))

        if resnet == 'resnet18':
            self.resnet = resnet18()
        elif resnet == 'resnet34':
            self.resnet = resnet34()
        else:
            raise RuntimeError(f'Unknown resnet type: {resnet}')

        self.adaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.linear = nn.Linear(512, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv3d(x)))  # batch x 64 x 29 x 56 x 56

        x = self.max_pool3d(x)  # batch x 64 x 29 x 28 x 28

        # transpose channel and depth
        x = x.transpose(1, 2)
        x = x.contiguous()

        # merge batch and depth
        batch, depth, channel, height, width = x.shape
        x = x.view(batch * depth, channel, height, width)

        x = self.resnet(x)  # batch*29 x 512 x 4 x 4

        x = self.adaptiveAvgPool2d(x).squeeze()  # batch*29 x 512

        x = self.bn2(self.linear(x))  # batch*29 x out_channels

        x = x.view(batch, depth, -1)  # batch x 29 x out_channels

        return x

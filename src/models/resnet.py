import torch.nn as nn
import torch.nn.functional as F


def resnet18():
    return ResNet(depths=[2, 2, 2, 2])


def resnet34():
    return ResNet(depths=[3, 4, 6, 3])


class ResNet(nn.Module):
    def __init__(self, depths, block_sizes=[64, 128, 256, 512]):
        super().__init__()

        self.model = nn.Sequential(*[
            ResidualLayer(inc, outc, depths[i], preactivation=(i > 0))
            for i, (inc,
                    outc) in enumerate(zip(block_sizes[:-1], block_sizes[1:]))
        ])

    def forward(self, x):
        return self.model(x)


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, preactivation=True):
        super().__init__()
        self.model = nn.Sequential(
            ResidualBlock(in_channels,
                          out_channels,
                          preactivation=preactivation), *[
                              ResidualBlock(out_channels, out_channels)
                              for _ in range(blocks - 1)
                          ])

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, preactivation=True):
        super().__init__()

        if preactivation:
            self.bn1 = nn.BatchNorm2d(in_channels)

        # spatial downsampling is done when the number of channels is increased
        stride = 2 if in_channels != out_channels else 1

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.shortcut_conv = nn.Conv2d(in_channels,
                                       out_channels,
                                       kernel_size=(1, 1),
                                       stride=stride,
                                       bias=False)

    def forward(self, x):
        # don't do bn and relu if we just did bn, relu and max_pool
        activation = F.relu(self.bn1(x)) if hasattr(self, 'bn1') else x

        conv = self.conv1(activation)
        residual = self.conv2(F.relu(self.bn2(conv)))

        if x.shape != residual.shape:
            x = self.shortcut_conv(x)

        return x + residual

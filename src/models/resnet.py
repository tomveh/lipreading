import torch.nn as nn
import torch.nn.functional as F


def resnet18():
    return ResNet(num_blocks=[2, 2, 2, 2])


def resnet34():
    return ResNet(num_blocks=[3, 4, 6, 3])


# note that there is no bottleneck so this only works for resnet18 and resnet34
class ResNet(nn.Module):
    def __init__(self, num_blocks, block_sizes=[64, 128, 256, 512]):
        super().__init__()

        self.model = nn.Sequential(
            ResidualLayer(64, 64, num_blocks[0], preactivation=False),
            ResidualLayer(64, 128, num_blocks[1], preactivation=True),
            ResidualLayer(128, 256, num_blocks[2], preactivation=True),
            ResidualLayer(256, 512, num_blocks[3], preactivation=True))

        # channels = enumerate(zip(block_sizes[:-1], block_sizes[1:]))

        # self.model = nn.Sequential(*[
        #     ResidualLayer(in_channels,
        #                   out_channels,
        #                   blocks=num_blocks[i],
        #                   preactivation=(i > 0))
        #     for i, (in_channels, out_channels) in channels
        # ])

    def forward(self, x):
        return self.model(x)


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, preactivation):
        super().__init__()
        self.model = nn.Sequential(
            ResidualBlock(in_channels,
                          out_channels,
                          preactivation=preactivation), *[
                              ResidualBlock(out_channels,
                                            out_channels,
                                            preactivation=True)
                              for _ in range(blocks - 1)
                          ])

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, preactivation):
        super().__init__()

        if preactivation:
            self.bn1 = nn.BatchNorm2d(in_channels)

        # spatial downsampling is done when the number of channels is increased
        stride = 2 if in_channels != out_channels else 1

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1 = _conv3x3(in_channels, out_channels, stride=stride)
        self.conv2 = _conv3x3(out_channels, out_channels, stride=1)
        self.shortcut_conv = _conv1x1(in_channels, out_channels, stride=stride)

    def forward(self, x):
        # don't do bn and relu if we just did bn, relu and max_pool
        activation = F.relu(self.bn1(x)) if hasattr(self, 'bn1') else x

        conv = self.conv1(activation)
        residual = self.conv2(F.relu(self.bn2(conv)))

        if x.shape != residual.shape:
            x = self.shortcut_conv(x)

        return x + residual


def _conv3x3(in_channels, out_channels, stride):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=(3, 3),
                     stride=stride,
                     padding=1,
                     bias=False)


def _conv1x1(in_channels, out_channels, stride):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=(1, 1),
                     stride=stride,
                     bias=False)

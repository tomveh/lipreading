import torch.nn as nn

from .resnet import resnet18

# class SpatioTempralFrontend(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv3d(1,
#                       64,
#                       kernel_size=(5, 7, 7),
#                       stride=(1, 2, 2),
#                       padding=(2, 3, 3),
#                       bias=False), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
#             nn.Dropout3d(p=0.1, inplace=True),
#             nn.MaxPool3d(kernel_size=(1, 3, 3),
#                          stride=(1, 2, 2),
#                          padding=(0, 1, 1)))


class VisualFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1,
                      64,
                      kernel_size=(5, 7, 7),
                      stride=(1, 2, 2),
                      padding=(2, 3, 3),
                      bias=False), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3),
                         stride=(1, 2, 2),
                         padding=(0, 1, 1)))

        self.resnet = nn.Sequential(resnet18(),
                                    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                    nn.Flatten(), nn.BatchNorm1d(512))
        self._init_weights()

    def _init_weights(self):
        for name, module in self.resnet.named_modules():
            if 'Conv2d' in module.__class__.__name__:
                nn.init.kaiming_uniform_(module.weight.data)

    def forward(self, x):
        x = self.conv3d(x)  # batch x 64 x depth x 56 x 56

        # transpose channel and depth dimensions
        x = x.transpose(1, 2).contiguous()

        batch, depth, channel, height, width = x.shape

        # merge batch and depth
        x = x.view(batch * depth, channel, height, width)

        x = self.resnet(x)  # batch*depth x 512

        x = x.view(batch, depth, -1)  # batch x depth x 512

        return x


class SpatioTemporalFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(1, 3, 3), stride=1),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        self.conv4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=(1, 3, 3), stride=1),
            nn.BatchNorm3d(512), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=(1, 2, 2), stride=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.squeeze().transpose(1, 2).contiguous()
        return x

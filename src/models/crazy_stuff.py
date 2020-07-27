import torch.nn as nn
import torchvision.models.video.resnet as torchvision_resnet


class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = Model()

        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        encoder_norm = nn.LayerNorm(512)
        self.t = nn.TransformerEncoder(encoder_layer,
                                       num_layers=2,
                                       norm=encoder_norm)

        self.linear = nn.Linear(512, 500)

    def forward(self, x):
        x = self.m(x)

        src = x.transpose(0, 1)

        out = self.t(src).transpose(0, 1)

        return self.linear(out.mean(dim=1))


def r2plus1d():
    # mod = VideoResNet2(block=torchvision_resnet.BasicBlock,
    #                    conv_makers=[Conv2Plus1DNoTemporal] * 4,
    #                    layers=[1, 1, 1, 1],
    #                    stem=torchvision_resnet.R2Plus1dStem)

    # mod.stem[0] = nn.Conv3d(1,
    #                         16,
    #                         kernel_size=(1, 7, 7),
    #                         stride=(1, 2, 2),
    #                         padding=(0, 3, 3),
    #                         bias=False)
    # mod.fc = nn.Linear(in_features=512, out_features=512)
    # mod.avgpool = nn.AdaptiveAvgPool3d(output_size=(None, 1, 1))

    # return mod

    return Model()


class VideoResNet2(torchvision_resnet.VideoResNet):
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.transpose(1, 2).flatten(2)
        x = self.fc(x)
        return x


class Conv2Plus1DNoTemporal(nn.Sequential):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):
        super(Conv2Plus1DNoTemporal, self).__init__(
            nn.Conv3d(in_planes,
                      midplanes,
                      kernel_size=(1, 3, 3),
                      stride=(1, stride, stride),
                      padding=(0, padding, padding),
                      bias=False), nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes,
                      out_planes,
                      kernel_size=(3, 1, 1),
                      stride=(1, 1, 1),
                      padding=(padding, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1,
                      45,
                      kernel_size=(1, 7, 7),
                      stride=(1, 2, 2),
                      padding=(0, 3, 3),
                      bias=False), nn.BatchNorm3d(45), nn.ReLU(inplace=True),
            nn.Conv3d(45,
                      64,
                      kernel_size=(3, 1, 1),
                      stride=(1, 1, 1),
                      padding=(1, 0, 0),
                      bias=False), nn.BatchNorm3d(64), nn.ReLU(inplace=True))

        self.l1 = nn.Sequential(
            nn.Conv3d(64,
                      64,
                      kernel_size=(1, 3, 3),
                      padding=(0, 1, 1),
                      stride=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            # nn.BatchNorm3d(64),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            # nn.BatchNorm3d(64),
            # nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(1, 1, 1), stride=(1, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.l2 = nn.Sequential(
            nn.Conv3d(128,
                      128,
                      kernel_size=(1, 3, 3),
                      padding=(0, 1, 1),
                      stride=(1, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            # nn.Conv3d(128, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            # nn.BatchNorm3d(128),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(128, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            # nn.BatchNorm3d(128),
            # nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(1, 1, 1), stride=(1, 2, 2)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

        self.l3 = nn.Sequential(
            nn.Conv3d(256,
                      256,
                      kernel_size=(1, 3, 3),
                      padding=(0, 1, 1),
                      stride=(1, 2, 2)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            # nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            # nn.BatchNorm3d(256),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            # nn.BatchNorm3d(256),
            # nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(1, 2, 2)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x) + self.shortcut1(x)
        x = self.l2(x) + self.shortcut2(x)
        x = self.l3(x) + self.shortcut3(x)

        x = self.avgpool(x).flatten(2)

        return x.transpose(1, 2)

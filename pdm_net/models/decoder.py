from torch import nn


class FrameDecoder(nn.Module):
    def __init__(self, base_channels, feature_channels):
        super().__init__()
        layers = []
        previous = 2 * feature_channels
        for channels in (base_channels * 4, base_channels * 2, base_channels, base_channels):
            layers.extend([nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                           nn.Conv2d(previous, channels, 3, padding=1),
                           nn.LeakyReLU(0.2, inplace=True)])
            previous = channels
        layers.extend([nn.Conv2d(previous, 3, 3, padding=1), nn.Sigmoid()])
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features)

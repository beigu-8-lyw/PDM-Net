from torch import nn


def widths(base, feature):
    return (base, base * 2, base * 4, feature)


class SpatialEncoder(nn.Module):
    """Shared per-frame appearance encoder; four spatial downsamplings."""

    def __init__(self, base_channels, feature_channels):
        super().__init__()
        layers = []
        previous = 3
        for channels in widths(base_channels, feature_channels):
            layers.extend([nn.Conv2d(previous, channels, 3, stride=2, padding=1),
                           nn.LeakyReLU(0.2, inplace=True)])
            previous = channels
        self.network = nn.Sequential(*layers)

    def forward(self, clip):
        batch, time, channels, height, width = clip.shape
        features = self.network(clip.reshape(batch * time, channels, height, width))
        return features.reshape(batch, time, *features.shape[1:])


class DynamicEncoder(nn.Module):
    """Engineering choice: 3D convolutions followed by temporal averaging.

    Temporal kernels learn ordered residual patterns; only space is strided.
    The paper does not specify this exact convolutional architecture.
    """

    def __init__(self, base_channels, feature_channels):
        super().__init__()
        layers = []
        previous = 3
        for channels in widths(base_channels, feature_channels):
            layers.extend([nn.Conv3d(previous, channels, 3, stride=(1, 2, 2), padding=1),
                           nn.LeakyReLU(0.2, inplace=True)])
            previous = channels
        self.network = nn.Sequential(*layers)

    def forward(self, residuals):
        return self.network(residuals.transpose(1, 2)).mean(dim=2)


class LongTermDynamicEncoder(DynamicEncoder):
    pass


class ShortTermDynamicEncoder(DynamicEncoder):
    pass

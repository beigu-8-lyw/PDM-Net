import torch
from torch import nn


class PrototypeFeatureAggregation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, max(channels // 4, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 4, 1), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, prototype_features, hidden, cell):
        attention = self.attention(torch.cat((prototype_features, cell), dim=1))
        return torch.cat((attention * prototype_features, hidden), dim=1)

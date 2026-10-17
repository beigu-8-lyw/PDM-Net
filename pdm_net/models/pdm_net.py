from dataclasses import dataclass

import torch
from torch import nn

from ..config import PDMConfig
from .aggregation import PrototypeFeatureAggregation
from .convlstm import StackedConvLSTM
from .decoder import FrameDecoder
from .encoders import SpatialEncoder, LongTermDynamicEncoder, ShortTermDynamicEncoder
from .prototype import DynamicPrototypeMemory, MemoryRead
from .residual import ResidualSequenceBuilder


@dataclass
class PDMOutput:
    prediction: torch.Tensor
    memory: MemoryRead


class PDMNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or PDMConfig()
        c = self.config
        self.residual = ResidualSequenceBuilder(c.residual_mode)
        self.spatial_encoder = SpatialEncoder(c.base_channels, c.feature_channels)
        self.long_encoder = LongTermDynamicEncoder(c.base_channels, c.feature_channels)
        self.short_encoder = ShortTermDynamicEncoder(c.base_channels, c.feature_channels)
        self.temporal = StackedConvLSTM(c.feature_channels, c.lstm_layers)
        self.memory = DynamicPrototypeMemory(c.num_prototypes, c.feature_channels)
        self.aggregation = PrototypeFeatureAggregation(c.feature_channels)
        self.decoder = FrameDecoder(c.base_channels, c.feature_channels)

    def _validate(self, clip, length):
        c = self.config
        if clip.ndim != 5 or tuple(clip.shape[1:]) != (length, 3, c.image_size, c.image_size):
            raise ValueError(f"Expected [B,{length},3,{c.image_size},{c.image_size}], got {tuple(clip.shape)}")
        if not clip.is_floating_point() or clip.shape[0] == 0:
            raise ValueError("Expected nonempty floating point clips in [0,1]")

    def _predict(self, short_clip, dynamic_features):
        recalled = self.memory(dynamic_features)
        hidden, cell = self.temporal(self.spatial_encoder(short_clip))
        prediction = self.decoder(self.aggregation(recalled.recalled, hidden, cell))
        return PDMOutput(prediction, recalled)

    def forward_long(self, short_clip, long_clip):
        self._validate(short_clip, self.config.short_frames)
        self._validate(long_clip, self.config.long_frames)
        if short_clip.shape[0] != long_clip.shape[0]:
            raise ValueError("Paired clips must share batch size")
        return self._predict(short_clip, self.long_encoder(self.residual(long_clip)))

    def forward(self, short_clip):
        """Inference / step 2: accepts ONLY the observed short clip."""
        self._validate(short_clip, self.config.short_frames)
        return self._predict(short_clip, self.short_encoder(self.residual(short_clip)))

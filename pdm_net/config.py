from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class PDMConfig:
    long_frames: int = 13
    short_frames: int = 9
    image_size: int = 256
    base_channels: int = 32
    feature_channels: int = 256
    lstm_layers: int = 2
    num_prototypes: int = 100
    learning_rate: float = 0.0002
    prototype_learning_rate: float = 0.0002
    loss_weight: float = 0.1
    score_weight: float = 0.6
    residual_mode: str = "signed"
    prediction_reduction: str = "mean"

    def __post_init__(self):
        integers = (self.long_frames, self.short_frames, self.image_size,
                    self.base_channels, self.feature_channels,
                    self.lstm_layers, self.num_prototypes)
        if any(type(v) is not int or v <= 0 for v in integers):
            raise ValueError("All dimensions and sequence lengths must be positive integers")
        if not 2 <= self.short_frames < self.long_frames:
            raise ValueError("Require 2 <= short_frames < long_frames")
        if self.image_size < 16 or self.image_size % 16:
            raise ValueError("image_size must be a positive multiple of 16")
        if self.num_prototypes < 2:
            raise ValueError("At least two prototypes are required")
        if self.residual_mode not in {"signed", "absolute"}:
            raise ValueError("residual_mode must be signed or absolute")
        if self.prediction_reduction not in {"mean", "sum_per_sample"}:
            raise ValueError("Invalid prediction_reduction")
        import math
        values = (self.learning_rate, self.prototype_learning_rate,
                  self.loss_weight, self.score_weight)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Hyperparameters must be finite")
        if min(self.learning_rate, self.prototype_learning_rate) <= 0:
            raise ValueError("Learning rates must be positive")
        if self.loss_weight < 0 or not 0 <= self.score_weight <= 1:
            raise ValueError("Invalid loss/score weight")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text()))

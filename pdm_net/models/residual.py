from torch import nn


class ResidualSequenceBuilder(nn.Module):
    def __init__(self, mode="signed"):
        super().__init__()
        if mode not in {"signed", "absolute"}:
            raise ValueError("Unknown residual mode")
        self.mode = mode

    def forward(self, frames):
        if frames.ndim != 5 or frames.shape[1] < 2:
            raise ValueError("Expected [B,T,C,H,W] with T >= 2")
        difference = frames[:, 1:] - frames[:, :-1]
        return difference.abs() if self.mode == "absolute" else difference

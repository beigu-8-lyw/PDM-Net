import torch
from torch import nn


def compactness(queries, nearest):
    return torch.linalg.vector_norm(queries - nearest, dim=-1).mean()


def dispersion(prototypes, indices):
    counts = torch.bincount(indices.detach().reshape(-1), minlength=len(prototypes))
    frequency = counts.to(prototypes.dtype) / indices.numel()
    centered = prototypes - prototypes.mean(dim=0, keepdim=True)
    # trace(sum p_i * outer(delta_i,delta_i)) = sum p_i * ||delta_i||^2
    return -(frequency * centered.square().sum(dim=-1)).sum()


class PDMLoss(nn.Module):
    def __init__(self, weight=0.1, prediction_reduction="mean"):
        super().__init__()
        if prediction_reduction not in {"mean", "sum_per_sample"}:
            raise ValueError("Invalid prediction reduction")
        self.weight = weight
        self.prediction_reduction = prediction_reduction

    def forward(self, output, target, prototypes, stage):
        if stage not in {"long", "short"}:
            raise ValueError("stage must be long or short")
        if output.prediction.shape != target.shape:
            raise ValueError("Prediction and target shapes must match exactly")
        difference = output.prediction - target
        per_element = difference.square() + difference.abs()
        prediction = (per_element.mean() if self.prediction_reduction == "mean"
                      else per_element.flatten(1).sum(dim=1).mean())
        compact = compactness(output.memory.queries, output.memory.nearest)
        spread = (dispersion(prototypes, output.memory.indices)
                  if stage == "long" else prediction.new_zeros(()))
        total = prediction + self.weight * (compact + spread)
        return {"total": total, "prediction": prediction,
                "compactness": compact, "dispersion": spread}

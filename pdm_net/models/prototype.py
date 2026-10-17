from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MemoryRead:
    recalled: torch.Tensor
    queries: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor
    nearest: torch.Tensor


class DynamicPrototypeMemory(nn.Module):
    """Cosine read (Eq. 3), query-normalized write (Eq. 4--5).

    Only the dispersion optimizer may differentiate the bank. Reads detach
    the bank, NOT the queries; compactness still trains both motion encoders.
    Explicit writes are performed only by training step 1, never forward().
    """

    def __init__(self, num_prototypes, channels):
        super().__init__()
        self.prototypes = nn.Parameter(F.normalize(torch.randn(num_prototypes, channels), dim=-1))

    def forward(self, features):
        batch, channels, height, width = features.shape
        queries = features.flatten(2).transpose(1, 2)
        bank = self.prototypes.detach()
        similarity = F.normalize(queries, dim=-1) @ F.normalize(bank, dim=-1).T
        weights = similarity.softmax(dim=-1)
        indices = similarity.argmax(dim=-1)
        recalled = (weights @ bank).transpose(1, 2).reshape(batch, channels, height, width)
        return MemoryRead(recalled, queries, weights, indices, bank[indices])

    @torch.no_grad()
    def normalize_(self):
        self.prototypes.copy_(F.normalize(self.prototypes, dim=-1))

    @torch.no_grad()
    def write(self, queries):
        """Treat all B*K positions as the update query set; keep empty items.

        Weights softmax over queries, then divide by their assigned-set max.
        Recompute assignments against the current (post-dispersion) bank.
        """
        queries = queries.detach().reshape(-1, self.prototypes.shape[1])
        if not len(queries) or not torch.isfinite(queries).all():
            raise ValueError("Memory write requires finite, nonempty queries")
        similarity = F.normalize(queries, dim=-1) @ F.normalize(self.prototypes, dim=-1).T
        assignments = similarity.argmax(dim=-1)
        write_weights = similarity.softmax(dim=0)
        updated = self.prototypes.clone()
        for index in range(len(updated)):
            selected = assignments == index
            if selected.any():
                weights = write_weights[selected, index]
                weights = weights / weights.max().clamp_min(1e-12)
                candidate = self.prototypes[index] + (weights[:, None] * queries[selected]).sum(dim=0)
                # An exactly cancelling sum must not destroy a prototype.
                if candidate.norm() > 1e-12:
                    updated[index] = F.normalize(candidate, dim=0)
        self.prototypes.copy_(updated)

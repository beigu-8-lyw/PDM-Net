from pathlib import Path

import torch

from ..losses import PDMLoss


class AlternatingTrainer:
    """Each batch: long SGD -> dispersion SGD -> explicit write -> short SGD.

    A single Adam owns shared predictor parameters across both stages. The
    separate bank Adam receives dispersion gradients only, and only in stage 1.
    This hybrid bank update is an explicit engineering resolution of the paper's
    ambiguous optimizer ownership; it is not claimed to match author code.
    """

    def __init__(self, model):
        self.model = model
        c = model.config
        self.criterion = PDMLoss(c.loss_weight, c.prediction_reduction)
        parameters = [p for name, p in model.named_parameters()
                      if name != "memory.prototypes"]
        self.optimizer = torch.optim.Adam(parameters, lr=c.learning_rate)
        self.prototype_optimizer = torch.optim.Adam(
            [model.memory.prototypes], lr=c.prototype_learning_rate)
        self.global_step = 0

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _step(self, batch, stage):
        self.model.train()
        short = batch["short_clip"].to(self.device)
        target = batch["target"].to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        self.prototype_optimizer.zero_grad(set_to_none=True)
        if stage == "long":
            output = self.model.forward_long(short, batch["long_clip"].to(self.device))
        elif stage == "short":
            output = self.model(short)
        else:
            raise ValueError("Unknown training stage")
        losses = self.criterion(output, target, self.model.memory.prototypes, stage)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"Non-finite {stage} loss")
        losses["total"].backward()
        # Fail before mutating parameters if any gradient is non-finite.
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Non-finite gradient: {name}")
        self.optimizer.step()
        if stage == "long":
            if self.model.config.loss_weight > 0:
                self.prototype_optimizer.step()
                self.model.memory.normalize_()
            self.model.memory.write(output.memory.queries)
        return {key: float(value.detach()) for key, value in losses.items()}

    def step_long(self, batch):
        return self._step(batch, "long")

    def step_short(self, batch):
        return self._step(batch, "short")

    def train_batch(self, batch):
        first = self.step_long(batch)
        second = self.step_short(batch)  # Fresh forward with updated shared network/bank.
        self.global_step += 1
        return {**{f"long/{k}": v for k, v in first.items()},
                **{f"short/{k}": v for k, v in second.items()}}

    def save_checkpoint(self, path, completed_epochs):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"format_version": 1, "config": self.model.config.to_dict(),
                 "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "prototype_optimizer": self.prototype_optimizer.state_dict(),
                 "global_step": self.global_step, "completed_epochs": completed_epochs,
                 "torch_rng": torch.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        if torch.backends.mps.is_available():
            state["mps_rng"] = torch.mps.get_rng_state()
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint version")
        if state["config"] != self.model.config.to_dict():
            raise ValueError("Checkpoint configuration differs from model configuration")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.prototype_optimizer.load_state_dict(state["prototype_optimizer"])
        self.global_step = state["global_step"]
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if torch.backends.mps.is_available() and "mps_rng" in state:
            torch.mps.set_rng_state(state["mps_rng"])
        return state["completed_epochs"]

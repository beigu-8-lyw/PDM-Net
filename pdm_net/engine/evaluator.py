from collections import defaultdict

import numpy as np
import torch


def minmax(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Scores must be finite and nonempty")
    span = values.max() - values.min()
    if span <= 1e-12:
        return np.zeros_like(values)
    return (values - values.min()) / span


def binary_auc(labels, scores):
    """Exact rank AUC, averaging ranks of tied scores; None for one class."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != scores.shape or not len(labels):
        raise ValueError("Expected equal nonempty one-dimensional labels/scores")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("AUC requires binary labels and finite scores")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    left = 0
    while left < len(scores):
        right = left + 1
        while right < len(scores) and sorted_scores[right] == sorted_scores[left]:
            right += 1
        ranks[order[left:right]] = (left + 1 + right) / 2
        left = right
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


class AnomalyScorer:
    def __init__(self, beta=0.6):
        if not 0 <= beta <= 1:
            raise ValueError("beta must be in [0,1]")
        self.beta = beta

    @torch.no_grad()
    def raw_scores(self, output, target):
        if output.prediction.shape != target.shape:
            raise ValueError("Target shape mismatch")
        mse = (output.prediction - target).square().flatten(1).mean(dim=1)
        psnr = -10 * torch.log10(mse.clamp_min(1e-12))
        matching = torch.linalg.vector_norm(
            output.memory.queries - output.memory.nearest, dim=-1).mean(dim=1)
        return psnr.cpu().numpy(), matching.cpu().numpy()

    def combine_video(self, psnr, matching):
        if np.shape(psnr) != np.shape(matching):
            raise ValueError("Raw score lengths differ")
        return self.beta * (1 - minmax(psnr)) + (1 - self.beta) * minmax(matching)


class Evaluator:
    def __init__(self, model):
        self.model = model
        self.scorer = AnomalyScorer(model.config.score_weight)

    @torch.no_grad()
    def run(self, loader):
        was_training = self.model.training
        self.model.eval()
        device = next(self.model.parameters()).device
        groups = defaultdict(list)
        try:
            for batch in loader:
                output = self.model(batch["short_clip"].to(device))
                psnr, matching = self.scorer.raw_scores(output, batch["target"].to(device))
                for i, video_id in enumerate(batch["video_id"]):
                    row = {"video_id": video_id, "target_index": int(batch["target_index"][i]),
                           "target_file": batch["target_file"][i], "psnr": float(psnr[i]),
                           "matching": float(matching[i])}
                    if "label" in batch:
                        row["label"] = int(batch["label"][i])
                    groups[video_id].append(row)
        finally:
            self.model.train(was_training)
        if not groups:
            raise ValueError("Empty evaluation loader")
        rows, video_auc = [], {}
        for video_id, items in sorted(groups.items()):
            items.sort(key=lambda item: item["target_index"])
            if len({item["target_index"] for item in items}) != len(items):
                raise ValueError(f"Duplicate target frame in {video_id}")
            scores = self.scorer.combine_video([r["psnr"] for r in items],
                                               [r["matching"] for r in items])
            for row, score in zip(items, scores):
                row["anomaly_score"] = float(score)
            if all("label" in row for row in items):
                video_auc[video_id] = binary_auc([r["label"] for r in items], scores)
            rows.extend(items)
        auc = (binary_auc([r["label"] for r in rows], [r["anomaly_score"] for r in rows])
               if all("label" in row for row in rows) else None)
        return rows, {"frame_auc": auc, "per_video_auc": video_auc,
                      "num_scored_frames": len(rows), "num_videos": len(groups)}

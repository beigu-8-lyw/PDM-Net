from bisect import bisect_right
from pathlib import Path
import re
import warnings

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def natural_key(path):
    return tuple((1, int(part)) if part.isdigit() else (0, part.lower())
                 for part in re.split(r"(\d+)", path.name))


class VideoClipDataset(Dataset):
    """Read root/video_id/frame files; indices are zero-based sorted positions.

    Training data must contain normal videos only. Test mode reads only the
    short prefix and target: hidden intermediate frames never enter the model.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, root, config, training=True, labels_dir=None, stride=1):
        self.root = Path(root)
        self.config = config
        self.training = training
        if type(stride) is not int or stride < 1:
            raise ValueError("stride must be a positive integer")
        self.stride = stride
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.videos = []
        self.ends = []
        self.labels = {}
        total = 0
        for directory in sorted((p for p in self.root.iterdir() if p.is_dir()), key=natural_key):
            frames = sorted((p for p in directory.iterdir()
                             if p.is_file() and p.suffix.lower() in self.EXTENSIONS), key=natural_key)
            if len(frames) <= config.long_frames:
                warnings.warn(f"Skipping {directory.name}: needs at least {config.long_frames + 1} frames")
                continue
            if labels_dir is not None:
                label_path = Path(labels_dir) / f"{directory.name}.npy"
                labels = np.load(label_path, allow_pickle=False)
                if labels.shape != (len(frames),) or not np.isin(labels, [0, 1]).all():
                    raise ValueError(f"{label_path}: expected {len(frames)} binary frame labels")
                if training and labels.any():
                    raise ValueError("Normal-only training cannot include anomalous labels")
                self.labels[directory.name] = labels
            windows = (len(frames) - config.long_frames - 1) // stride + 1
            total += windows
            self.videos.append((directory.name, frames))
            self.ends.append(total)
        if not self.videos:
            raise ValueError(f"No valid videos under {self.root}; use root/video_id/frame.png")

    def __len__(self):
        return self.ends[-1]

    def _read(self, path):
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.config.image_size, self.config.image_size), Image.Resampling.BILINEAR)
            array = np.array(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        video_index = bisect_right(self.ends, index)
        previous_end = self.ends[video_index - 1] if video_index else 0
        start = (index - previous_end) * self.stride
        video_id, frames = self.videos[video_index]
        target_index = start + self.config.long_frames
        count = self.config.long_frames if self.training else self.config.short_frames
        clip = torch.stack([self._read(path) for path in frames[start:start + count]])
        sample = {"short_clip": clip[:self.config.short_frames],
                  "target": self._read(frames[target_index]),
                  "video_id": video_id, "target_index": target_index,
                  "target_file": frames[target_index].name}
        if self.training:
            sample["long_clip"] = clip
        if video_id in self.labels:
            sample["label"] = int(self.labels[video_id][target_index])
        return sample

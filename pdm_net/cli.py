import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import PDMConfig, PDMNet
from .data import VideoClipDataset
from .engine import AlternatingTrainer, Evaluator


def resolve_device(name):
    if name == "auto":
        # Conv3d support on MPS differs across PyTorch/macOS releases.
        # Use CPU unless CUDA is present; MPS is an explicit opt-in.
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def read_state(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint version")
    return state


def train(args):
    torch.manual_seed(args.seed)
    if args.resume:
        config = PDMConfig(**read_state(args.resume)["config"])
        if args.config and PDMConfig.from_json(args.config) != config:
            raise ValueError("Resume checkpoint and --config differ")
    else:
        config = PDMConfig.from_json(args.config) if args.config else PDMConfig()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "latest.pt"
    if checkpoint.exists() and not args.resume:
        raise FileExistsError("Output already has latest.pt; use --resume or a different --output")
    dataset = VideoClipDataset(args.data, config, training=True, stride=args.stride)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    model = PDMNet(config).to(resolve_device(args.device))
    trainer = AlternatingTrainer(model)
    start_epoch = trainer.load_checkpoint(args.resume) if args.resume else 0
    if args.epochs <= start_epoch:
        raise ValueError("--epochs must exceed the checkpoint's completed epoch count")
    (destination / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    print(json.dumps({"device": str(trainer.device), "training_windows": len(dataset),
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "start_epoch": start_epoch}), flush=True)
    for epoch in range(start_epoch, args.epochs):
        totals = {}
        count = 0
        for batch_index, batch in enumerate(loader):
            metrics = trainer.train_batch(batch)
            batch_size = len(batch["short_clip"])
            count += batch_size
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
            if (batch_index + 1) % args.log_every == 0:
                print(json.dumps({"epoch": epoch + 1, "batch": batch_index + 1, **metrics}), flush=True)
        record = {"epoch": epoch + 1, "global_step": trainer.global_step,
                  **{key: value / count for key, value in totals.items()}}
        trainer.save_checkpoint(checkpoint, completed_epochs=epoch + 1)
        with (destination / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


def evaluate(args):
    state = read_state(args.checkpoint)
    config = PDMConfig(**state["config"])
    model = PDMNet(config).to(resolve_device(args.device))
    model.load_state_dict(state["model"])
    dataset = VideoClipDataset(args.data, config, training=False, labels_dir=args.labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    rows, metrics = Evaluator(model).run(loader)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


def smoke(args):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    config = PDMConfig(long_frames=5, short_frames=3, image_size=32,
                       base_channels=4, feature_channels=8, num_prototypes=5)
    model = PDMNet(config).to(resolve_device(args.device))
    trainer = AlternatingTrainer(model)
    long_clip = torch.rand(2, 5, 3, 32, 32)
    batch = {"long_clip": long_clip, "short_clip": long_clip[:, :3],
             "target": torch.rand(2, 3, 32, 32)}
    metrics = trainer.train_batch(batch)
    model.eval()
    with torch.no_grad():
        output = model(batch["short_clip"].to(trainer.device))
    print(json.dumps({"device": str(trainer.device), "prediction_shape": list(output.prediction.shape),
                      "parameters": sum(p.numel() for p in model.parameters()), **metrics}, indent=2))


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description="PDM-Net research baseline")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke_parser = commands.add_parser("smoke", help="Tiny synthetic two-step training check")
    smoke_parser.add_argument("--device", default="cpu")
    smoke_parser.set_defaults(action=smoke)
    train_parser = commands.add_parser("train", help="Train on normal-only frame folders")
    eval_parser = commands.add_parser("evaluate", help="Offline per-video anomaly scores")
    for command in (train_parser, eval_parser):
        command.add_argument("--data", required=True, help="Root of video/frame folders")
        command.add_argument("--output", required=True)
        command.add_argument("--batch-size", type=positive_int, default=1)
        command.add_argument("--workers", type=int, choices=range(0, 65), default=0)
        command.add_argument("--device", default="auto", help="auto (CUDA/CPU), cpu, cuda, or mps")
    train_parser.add_argument("--config")
    train_parser.add_argument("--epochs", type=positive_int, default=50, help="Total epochs, including resumed epochs")
    train_parser.add_argument("--stride", type=positive_int, default=1)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--resume")
    train_parser.add_argument("--log-every", type=positive_int, default=20)
    train_parser.set_defaults(action=train)
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--labels", help="Optional video_id.npy arrays: one binary label per frame")
    eval_parser.set_defaults(action=evaluate)
    args = parser.parse_args()
    args.action(args)

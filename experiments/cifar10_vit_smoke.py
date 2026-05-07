from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from newton_muon import Muon, NewtonMuon


class TinyViT(nn.Module):
    def __init__(self, dim: int = 128, depth: int = 4, heads: int = 4, patch: int = 4):
        super().__init__()
        self.patch = patch
        self.patch_proj = nn.Linear(3 * patch * patch, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, (32 // patch) ** 2 + 1, dim))
        enc = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(enc, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).flatten(1, 2).flatten(2)
        x = self.patch_proj(x)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        x = self.blocks(x)
        return self.head(self.norm(x[:, 0]))


def split_params(model: nn.Module):
    matrix, other = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and not name.endswith("head.weight"):
            matrix.append(p)
        else:
            other.append(p)
    return matrix, other


def build_optimizer(name: str, model: nn.Module, args: argparse.Namespace):
    matrix, other = split_params(model)
    if name == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=args.adam_lr, weight_decay=args.weight_decay)]
    if name == "muon":
        return [
            Muon(matrix, lr=args.muon_lr, momentum=args.momentum, weight_decay=args.weight_decay),
            torch.optim.AdamW(other, lr=args.adam_lr, weight_decay=args.weight_decay),
        ]
    if name == "newton_muon":
        opt = NewtonMuon(
            matrix,
            lr=args.newton_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            beta=args.beta,
            ridge=args.ridge,
            refresh_interval=args.refresh_interval,
            max_samples=args.max_samples,
        )
        opt.attach(model)
        return [opt, torch.optim.AdamW(other, lr=args.adam_lr, weight_decay=args.weight_decay)]
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=["adamw", "muon", "newton_muon"], default="newton_muon")
    parser.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--adam-lr", type=float, default=3e-4)
    parser.add_argument("--muon-lr", type=float, default=2e-3)
    parser.add_argument("--newton-lr", type=float, default=2e-3)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--ridge", type=float, default=0.2)
    parser.add_argument("--refresh-interval", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    ds = datasets.CIFAR10(str(args.data_dir), train=True, transform=transform, download=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.device == "cuda")

    torch.manual_seed(0)
    model = TinyViT().to(args.device)
    opts = build_optimizer(args.optimizer, model, args)
    criterion = nn.CrossEntropyLoss()

    model.train()
    t0 = time.perf_counter()
    step = 0
    rows = []
    while step < args.steps:
        for x, y in loader:
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            for opt in opts:
                opt.step()
            step += 1
            elapsed = time.perf_counter() - t0
            rows.append(
                {
                    "optimizer": args.optimizer,
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "wall_time_s": elapsed,
                    "images_per_s": step * args.batch_size / max(elapsed, 1e-9),
                }
            )
            if step % 20 == 0 or step == 1:
                print(f"step:{step}/{args.steps} loss:{loss.item():.4f} time:{elapsed:.1f}s img_s:{step * args.batch_size / max(elapsed, 1e-9):.1f}")
            if step >= args.steps:
                break

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        exists = args.out.exists()
        with args.out.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["optimizer", "step", "loss", "wall_time_s", "images_per_s"])
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
        print(args.out)


if __name__ == "__main__":
    main()

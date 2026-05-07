from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn

from newton_muon import Muon, NewtonMuon


def make_batch(batch_size: int, d_in: int, d_out: int, condition: float, device: str):
    scales = torch.logspace(0, torch.log10(torch.tensor(condition)), d_in, device=device)
    x = torch.randn(batch_size, d_in, device=device) * scales
    w_star = torch.randn(d_out, d_in, device=device) / (d_in**0.5)
    y = x @ w_star.T
    return x, y


def build_optimizer(name: str, model: nn.Module, args: argparse.Namespace):
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.adam_lr, weight_decay=0.0, fused=args.device == "cuda")
    if name == "muon":
        return Muon(model.parameters(), lr=args.muon_lr, momentum=args.momentum)
    if name == "newton_muon":
        opt = NewtonMuon(
            model.parameters(),
            lr=args.newton_lr,
            momentum=args.momentum,
            beta=args.beta,
            ridge=args.ridge,
            refresh_interval=args.refresh_interval,
            max_samples=args.max_samples,
        )
        opt.attach(model)
        return opt
    raise ValueError(name)


def run_one(name: str, args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    torch.manual_seed(args.seed)
    model = nn.Linear(args.d_in, args.d_out, bias=False, device=args.device)
    x, y = make_batch(args.batch_size, args.d_in, args.d_out, args.condition, args.device)
    opt = build_optimizer(name, model, args)

    rows = []
    if args.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(args.steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = (model(x) - y).square().mean()
        if step == args.steps:
            break
        loss.backward()
        opt.step()

        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "optimizer": name,
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "wall_time_s": elapsed,
                "examples_per_s": (step + 1) * args.batch_size / max(elapsed, 1e-9),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d-in", type=int, default=128)
    parser.add_argument("--d-out", type=int, default=128)
    parser.add_argument("--condition", type=float, default=64.0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--adam-lr", type=float, default=3e-3)
    parser.add_argument("--muon-lr", type=float, default=2e-2)
    parser.add_argument("--newton-lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--ridge", type=float, default=0.2)
    parser.add_argument("--refresh-interval", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--out", type=Path, default=Path("reports/toy_quadratic_metrics.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for name in ["adamw", "muon", "newton_muon"]:
        rows = run_one(name, args)
        all_rows.extend(rows)
        print(f"{name}: final_loss={rows[-1]['loss']:.6f} time={rows[-1]['wall_time_s']:.3f}s")

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["optimizer", "step", "loss", "wall_time_s", "examples_per_s"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(args.out)


if __name__ == "__main__":
    main()

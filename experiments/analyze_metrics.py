from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--target-loss", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    if "optimizer" not in df:
        raise ValueError("CSV must contain an optimizer column")

    rows = []
    for opt, part in df.groupby("optimizer", sort=False):
        final = part.iloc[-1]
        row = {
            "optimizer": opt,
            "final_step": int(final["step"]),
            "final_loss": float(final["loss"]),
            "wall_time_s": float(final["wall_time_s"]),
        }
        throughput_cols = [c for c in ["examples_per_s", "images_per_s", "tokens_per_s"] if c in final.index]
        if throughput_cols:
            row["throughput"] = float(final[throughput_cols[0]])
            row["throughput_col"] = throughput_cols[0]
        if args.target_loss is not None:
            hit = part[part["loss"] <= args.target_loss]
            row["target_loss"] = args.target_loss
            row["time_to_target_s"] = float(hit.iloc[0]["wall_time_s"]) if not hit.empty else float("nan")
            row["steps_to_target"] = int(hit.iloc[0]["step"]) if not hit.empty else -1
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = args.out_dir / f"{args.csv.stem}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(summary_path)

    plt.figure(figsize=(7, 4.5))
    for opt, part in df.groupby("optimizer", sort=False):
        plt.plot(part["step"], part["loss"], label=opt)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    loss_plot = args.out_dir / f"{args.csv.stem}_loss.png"
    plt.savefig(loss_plot, dpi=160)
    print(loss_plot)

    plt.figure(figsize=(7, 4.5))
    for opt, part in df.groupby("optimizer", sort=False):
        plt.plot(part["wall_time_s"], part["loss"], label=opt)
    plt.xlabel("wall-clock seconds")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    time_plot = args.out_dir / f"{args.csv.stem}_time.png"
    plt.savefig(time_plot, dpi=160)
    print(time_plot)


if __name__ == "__main__":
    main()

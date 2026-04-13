#!/usr/bin/env python3
"""
backtest_slope_stopping.py — Simulate slope-based early stopping on a recorded WandB run.

Usage:
    python scripts/backtest_slope_stopping.py <run_id> [options]

The run_id can be a bare id (e.g. intergalactic_leakey_...) or a full path
(entity/project/run_id).  The WandB entity/project are taken from the run_id
if provided, otherwise from --entity / --project (defaults: tansey-lab/multitme).

Requires WANDB_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from multitme.model import check_loss_slope_convergence

# ---------------------------------------------------------------------------
# Stopping simulation
# ---------------------------------------------------------------------------


def find_stop_epoch(losses: list[float], window: int, threshold: float) -> int | None:
    """Return the first epoch at which check_loss_slope_convergence fires."""
    for epoch in range(len(losses)):
        if check_loss_slope_convergence(losses[: epoch + 1], window, threshold):
            return epoch
    return None


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def parameter_grid(losses: list[float], windows: list[int], thresholds: list[float]) -> None:
    n = len(losses)
    final = losses[-1]
    initial = losses[0]
    total_drop = final - initial

    print(
        f"\n{'Window':>8}  {'Threshold':>12}  {'Stop epoch':>12}  {'% trained':>10}  "
        f"{'Loss at stop':>14}  {'Final loss':>12}  {'Loss recovered':>16}"
    )
    print("-" * 98)

    for window in windows:
        for threshold in thresholds:
            stop = find_stop_epoch(losses, window, threshold)
            if stop is not None:
                loss_at_stop = losses[stop]
                pct_trained = 100.0 * stop / n
                pct_recovered = (
                    100.0 * (loss_at_stop - initial) / total_drop
                    if total_drop != 0
                    else float("nan")
                )
                print(
                    f"{window:>8}  {threshold:>12.0e}  {stop:>12}  {pct_trained:>9.1f}%  "
                    f"{loss_at_stop:>14.1f}  {final:>12.1f}  {pct_recovered:>15.1f}%"
                )
            else:
                print(f"{window:>8}  {threshold:>12.0e}  {'never':>12}")


def make_plot(
    losses: list[float],
    windows: list[int],
    thresholds: list[float],
    run_name: str,
    out_path: Path,
) -> None:
    epochs = list(range(len(losses)))

    # One subplot per window size; markers for each threshold
    n_windows = len(windows)
    fig, axes = plt.subplots(n_windows, 1, figsize=(12, 4 * n_windows), sharex=True)
    if n_windows == 1:
        axes = [axes]

    # Colour cycle for thresholds
    cmap = plt.get_cmap("tab10")
    threshold_colors = {t: cmap(i) for i, t in enumerate(thresholds)}

    for ax, window in zip(axes, windows, strict=False):
        ax.plot(epochs, losses, color="steelblue", linewidth=1.5, label="total loss", zorder=1)

        for threshold in thresholds:
            stop = find_stop_epoch(losses, window, threshold)
            if stop is not None:
                color = threshold_colors[threshold]
                ax.axvline(stop, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
                ax.scatter(
                    [stop],
                    [losses[stop]],
                    color=color,
                    zorder=5,
                    s=60,
                    label=f"thresh={threshold:.0e}  epoch {stop}",
                )

        ax.set_title(f"window = {window}", fontsize=11)
        ax.set_ylabel("total loss")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("epoch")
    fig.suptitle(f"Slope-stopping backtest\n{run_name}", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out_path}")


def epoch_detail(losses: list[float], start: int = 0, stride: int = 5) -> None:
    print(f"\n{'Epoch':>7}  {'Loss':>14}  {'Delta':>12}  {'Rel delta':>12}")
    print("-" * 52)
    prev = losses[max(start - 1, 0)]
    for i in range(start, len(losses), stride):
        delta = losses[i] - prev
        rel = delta / abs(losses[i]) if losses[i] != 0.0 else float("nan")
        print(f"{i:>7}  {losses[i]:>14.1f}  {delta:>12.1f}  {rel:>12.6f}")
        prev = losses[i]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Backtest slope-based early stopping against a WandB training run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "run_id",
        help="WandB run id (bare id or entity/project/run_id)",
    )
    parser.add_argument("--entity", default="tansey-lab", help="WandB entity (if not in run_id)")
    parser.add_argument("--project", default="multitme", help="WandB project (if not in run_id)")
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[5, 10, 20],
        metavar="W",
        help="Slope window sizes to test",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[1e-3, 5e-4, 1e-4, 5e-5],
        metavar="T",
        help="Normalised-slope thresholds to test",
    )
    parser.add_argument(
        "--detail-start",
        type=int,
        default=None,
        metavar="EPOCH",
        help="Print per-epoch detail table starting at this epoch (default: last 25%% of run)",
    )
    parser.add_argument(
        "--detail-stride",
        type=int,
        default=5,
        metavar="N",
        help="Print every Nth epoch in the detail table",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        metavar="PATH",
        help="Save plot to this path (e.g. slope_backtest.png). Defaults to <run_id>_slope_backtest.png",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args(argv)

    # Build full run path
    run_path = args.run_id
    if run_path.count("/") < 2:
        run_path = f"{args.entity}/{args.project}/{run_path}"

    try:
        import wandb
    except ImportError:
        sys.exit("wandb is not installed. Run: pip install wandb")

    api = wandb.Api()
    print(f"Fetching run: {run_path}")
    run = api.run(run_path)
    hist = run.history(keys=["epoch", "train/total"], pandas=False)

    if not hist:
        sys.exit("No epoch loss history found in this run (expected 'train/total' key).")

    losses = [row["train/total"] for row in hist]
    n = len(losses)

    print(f"\nRun:        {run.name}")
    print(f"Epochs:     {n}")
    print(f"Loss range: {losses[0]:.1f}  ->  {losses[-1]:.1f}")

    parameter_grid(losses, windows=args.windows, thresholds=args.thresholds)

    detail_start = args.detail_start if args.detail_start is not None else max(0, n * 3 // 4)
    print(f"\nPer-epoch detail (epoch {detail_start} onward, stride {args.detail_stride}):")
    epoch_detail(losses, start=detail_start, stride=args.detail_stride)

    if not args.no_plot:
        plot_path = (
            Path(args.plot)
            if args.plot
            else Path(f"{args.run_id.split('/')[-1]}_slope_backtest.png")
        )
        make_plot(
            losses,
            windows=args.windows,
            thresholds=args.thresholds,
            run_name=run.name,
            out_path=plot_path,
        )


if __name__ == "__main__":
    main()

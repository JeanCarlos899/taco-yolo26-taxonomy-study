from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

from pipeline_utils import ROOT, TAXONOMIES


SEEDS = (42, 123, 2026)


def epoch_status(path: Path) -> str:
    if not path.exists():
        return "waiting"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return "initialized"
    epoch = int(float(rows[-1]["epoch"])) + 1
    best = path.parent / "weights" / "best.pt"
    metadata = path.parent / "experiment_config.json"
    complete = False
    if metadata.exists():
        with metadata.open(encoding="utf-8") as handle:
            complete = "training_time_seconds" in json.load(handle)
    return (f"complete at epoch {epoch}" if complete else f"epoch {epoch}/60") + ("; checkpoint available" if best.exists() else "")


def render() -> None:
    print("Paper experiment status\n")
    for seed in SEEDS:
        base = ROOT / "runs" if seed == 42 else ROOT / "runs" / "seeds" / str(seed)
        for taxonomy in TAXONOMIES:
            results = base / taxonomy / "results.csv"
            evaluated = (ROOT / "results" if seed == 42 else ROOT / "results" / "seeds" / str(seed)) / f"{taxonomy}_test_metrics.json"
            state = epoch_status(results)
            if evaluated.exists():
                state += "; evaluated"
            print(f"seed {seed:4d} | {taxonomy:8s} | {state}")
    print("\nAnalyses")
    for name in ("fine_hierarchical_evaluation.csv", "bootstrap_comparisons.csv",
                 "fine_error_decomposition.csv", "multiseed_summary.csv"):
        path = ROOT / "results" / name
        print(f"{name:36s} {'ready' if path.exists() else 'waiting'}")
    bootstrap_config = ROOT / "results" / "bootstrap_config.json"
    if bootstrap_config.exists():
        with bootstrap_config.open(encoding="utf-8") as handle:
            print(f"bootstrap iterations on disk: {json.load(handle).get('iterations')}")
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        print(f"\nGPU: {gpu}")
    except (OSError, subprocess.CalledProcessError):
        pass
    print(f"\nLogs: {ROOT / 'logs'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show or continuously watch the paper experiment.")
    parser.add_argument("--watch", action="store_true", help="Refresh continuously until Ctrl+C.")
    parser.add_argument("--interval", type=float, default=10.0, help="Refresh interval in seconds (default: 10).")
    args = parser.parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval must be greater than zero")
    try:
        while True:
            if args.watch:
                os.system("cls" if os.name == "nt" else "clear")
                print(time.strftime("Updated at %Y-%m-%d %H:%M:%S"))
            render()
            if not args.watch:
                break
            print(f"\nRefreshing every {args.interval:g}s. Press Ctrl+C to stop watching; experiments keep running.")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped. Background experiments were not interrupted.")


if __name__ == "__main__":
    main()

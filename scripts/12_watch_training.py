from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from pipeline_utils import ROOT, TAXONOMIES, config


def latest_epoch(taxonomy: str) -> dict[str, str] | None:
    path = ROOT / "runs" / taxonomy / "results.csv"
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return rows[-1] if rows else None
    except (OSError, csv.Error):
        return None


def number(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "nan").strip())


def snapshot() -> tuple[str, list[str]]:
    status_path = ROOT / "results" / "full_run_status.json"
    state = "starting"
    if status_path.exists():
        try:
            state = json.loads(status_path.read_text(encoding="utf-8"))["state"]
        except (OSError, json.JSONDecodeError, KeyError):
            state = "unknown"
    maximum = config()["epochs"]
    lines = [f"Pipeline state: {state}"]
    for taxonomy in TAXONOMIES:
        row = latest_epoch(taxonomy)
        best = ROOT / "runs" / taxonomy / "weights" / "best.pt"
        if row is None:
            lines.append(f"{taxonomy:8s} | waiting")
            continue
        lines.append(
            f"{taxonomy:8s} | epoch {int(number(row, 'epoch')):02d}/{maximum} | "
            f"P {number(row, 'metrics/precision(B)'):.4f} | "
            f"R {number(row, 'metrics/recall(B)'):.4f} | "
            f"mAP50 {number(row, 'metrics/mAP50(B)'):.4f} | "
            f"mAP50-95 {number(row, 'metrics/mAP50-95(B)'):.4f} | "
            f"best.pt {'yes' if best.exists() else 'no'}"
        )
    return state, lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean epoch-level monitor without ANSI progress bars.")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    previous = None
    while True:
        state, lines = snapshot()
        rendered = "\n".join(lines)
        if rendered != previous:
            print(f"\n[{time.strftime('%H:%M:%S')}]\n{rendered}", flush=True)
            previous = rendered
        if args.once or state in {"complete", "failed"}:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

import numpy as np

from pipeline_utils import ROOT, TAXONOMIES, config, load_json, save_csv


METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


def result_root(seed: int) -> object:
    return ROOT / "results" if seed == config()["seed"] else ROOT / "results" / "seeds" / str(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed seed-specific Ultralytics metrics.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    args = parser.parse_args()
    raw = []
    for seed in args.seeds:
        for taxonomy in TAXONOMIES:
            path = result_root(seed) / f"{taxonomy}_test_metrics.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing evaluation for seed {seed}: {path}")
            native = load_json(path)["ultralytics"]
            row = {"seed": seed, "taxonomy": taxonomy}
            row.update({name: float(native[key]) for name, key in METRICS.items()})
            p, r = row["precision"], row["recall"]
            row["f1"] = 2 * p * r / (p + r) if p + r else 0.0
            raw.append(row)
    save_csv(ROOT / "results" / "multiseed_raw.csv", raw, list(raw[0]))
    summary = []
    for taxonomy in TAXONOMIES:
        rows = [row for row in raw if row["taxonomy"] == taxonomy]
        item = {"taxonomy": taxonomy, "num_seeds": len(rows)}
        for metric in (*METRICS, "f1"):
            values = np.asarray([row[metric] for row in rows], dtype=float)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(item)
    save_csv(ROOT / "results" / "multiseed_summary.csv", summary, list(summary[0]))
    print(*summary, sep="\n")


if __name__ == "__main__":
    main()

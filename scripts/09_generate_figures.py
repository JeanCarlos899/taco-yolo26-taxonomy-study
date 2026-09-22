from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_utils import ROOT, TAXONOMIES, load_json


def save(fig, name: str) -> None:
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    documents = [load_json(ROOT / "results" / f"{t}_test_metrics.json") for t in TAXONOMIES]
    metrics = []
    for document in documents:
        native = document["ultralytics"]
        precision = native["metrics/precision(B)"]; recall = native["metrics/recall(B)"]
        metrics.append({"map50": native["metrics/mAP50(B)"], "map50_95": native["metrics/mAP50-95(B)"], "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0})
    labels = ["Fine", "Material", "Binary"]; keys = ["map50", "map50_95", "f1"]; display = ["AP50", "mAP50–95", "F1"]
    x = np.arange(len(labels)); width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for index, (key, title) in enumerate(zip(keys, display)):
        ax.bar(x + (index - 1) * width, [row[key] for row in metrics], width, label=title)
    ax.set_xticks(x, labels); ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.legend(); ax.grid(axis="y", alpha=.25)
    save(fig, "02_taxonomy_comparison")

    aware = documents[0]["controlled_evaluator"]; agnostic = load_json(ROOT / "results" / "fine_class_agnostic_metrics.json")
    keys = ["precision", "recall", "f1", "ap50", "map50_95"]
    x = np.arange(len(keys)); fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - .18, [aware[k] for k in keys], .36, label="Class-aware")
    ax.bar(x + .18, [agnostic[k] for k in keys], .36, label="Class-agnostic")
    ax.set_xticks(x, ["Precision", "Recall", "F1", "AP50", "mAP50–95"]); ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.legend(); ax.grid(axis="y", alpha=.25)
    save(fig, "03_fine_aware_vs_agnostic")

    groups = pd.read_csv(ROOT / "results" / "long_tail_groups.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.0)); x = np.arange(len(groups)); width = .27
    for index, (key, title) in enumerate((("mean_ap50", "Mean AP50"), ("mean_ap50_95", "Mean AP50–95"), ("mean_recall", "Mean recall"))):
        ax.bar(x + (index - 1) * width, groups[key], width, label=title)
    ax.set_xticks(x, groups["frequency_group"]); ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.legend(); ax.grid(axis="y", alpha=.25)
    save(fig, "04_long_tail_performance")


if __name__ == "__main__":
    main()

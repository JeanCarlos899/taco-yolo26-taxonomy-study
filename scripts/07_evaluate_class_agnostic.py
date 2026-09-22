from __future__ import annotations

from pathlib import Path

from evaluation_utils import evaluate_predictions, semantic_confusions
from pipeline_utils import ROOT, TAXONOMIES, load_json, save_csv, save_json


def main() -> None:
    summaries = []
    for taxonomy in TAXONOMIES:
        predictions = ROOT / "results" / "predictions" / f"{taxonomy}.json"
        if not predictions.exists():
            raise FileNotFoundError(f"Run scripts/06_evaluate.py first: {predictions}")
        aware, per_class = evaluate_predictions(taxonomy, predictions, agnostic=False)
        metrics_path = ROOT / "results" / f"{taxonomy}_test_metrics.json"
        existing = load_json(metrics_path)
        existing["controlled_evaluator"] = aware
        save_json(metrics_path, existing)
        save_csv(ROOT / "results" / f"{taxonomy}_class_metrics.csv", per_class, list(per_class[0]))
        agnostic, _ = evaluate_predictions(taxonomy, predictions, agnostic=True)
        summaries.append(agnostic)
        save_json(ROOT / "results" / f"{taxonomy}_class_agnostic_metrics.json", agnostic)
    save_csv(ROOT / "results" / "class_agnostic_summary.csv", summaries, list(summaries[0]))
    confusions = semantic_confusions(ROOT / "results" / "predictions" / "fine.json")
    save_csv(ROOT / "results" / "fine_semantic_confusions.csv", confusions, ["ground_truth", "predicted", "count"])
    print(*summaries, sep="\n")


if __name__ == "__main__":
    main()

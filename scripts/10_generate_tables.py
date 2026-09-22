from __future__ import annotations

from pipeline_utils import ROOT, TAXONOMIES, load_json, save_csv, taxonomy_names, coco


def main() -> None:
    rows = []
    for taxonomy in TAXONOMIES:
        document = load_json(ROOT / "results" / f"{taxonomy}_test_metrics.json")
        native = document["ultralytics"]
        precision = native["metrics/precision(B)"]
        recall = native["metrics/recall(B)"]
        metadata = load_json(ROOT / "runs" / taxonomy / "experiment_config.json")
        rows.append({
            "taxonomy": taxonomy, "num_classes": len(taxonomy_names(coco(), taxonomy)),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "map50": native["metrics/mAP50(B)"], "map50_95": native["metrics/mAP50-95(B)"],
            "training_time_seconds": metadata.get("training_time_seconds"),
        })
    save_csv(ROOT / "results" / "summary.csv", rows, list(rows[0]))
    aware = load_json(ROOT / "results" / "fine_test_metrics.json")["controlled_evaluator"]
    agnostic = load_json(ROOT / "results" / "fine_class_agnostic_metrics.json")
    comparison = [{"evaluation": label, **{key: values[key] for key in ("precision", "recall", "f1", "ap50", "map50_95")}} for label, values in (("Fine class-aware", aware), ("Fine class-agnostic", agnostic))]
    save_csv(ROOT / "results" / "fine_aware_vs_agnostic.csv", comparison, list(comparison[0]))
    print(*rows, sep="\n")


if __name__ == "__main__":
    main()

"""Repeat bootstrap and error decomposition for every trained seed.

The optimized bootstrap caches one-to-one matches and only reapplies image
multiplicities. A validation guard compares it with the original evaluator.
"""
from __future__ import annotations

import argparse
import importlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation_utils import class_at_iou, iou
from pipeline_utils import ROOT, coco, load_json, save_csv, save_json, taxonomy_names


analysis = importlib.import_module("14_strengthen_analysis")
SEEDS = (42, 123, 2026)
METRICS = ("ap50", "map50_95", "f1")
COMPARISONS = {
    "fine_direct_minus_material_direct": ("fine_direct", "material_direct"),
    "fine_as_material_minus_fine_direct": ("fine_as_material", "fine_direct"),
    "fine_as_binary_minus_fine_direct": ("fine_as_binary", "fine_direct"),
}


def result_root(seed: int) -> Path:
    return ROOT / "results" if seed == 42 else ROOT / "results/seeds" / str(seed)


def configurations(seed: int):
    root = result_root(seed)
    return {
        "fine_direct": ("fine", root / "predictions/fine.json"),
        "material_direct": ("material", root / "predictions/material.json"),
        "fine_as_material": ("material", root / "predictions/fine_as_material.json"),
        "fine_as_binary": ("binary", root / "predictions/fine_as_binary.json"),
    }


def prepare_evaluator(taxonomy: str, predictions_path: Path):
    gt = analysis.ground_truth(taxonomy)
    image_names = list(gt)
    image_index = {name: index for index, name in enumerate(image_names)}
    predictions = load_json(predictions_path)
    class_count = len(taxonomy_names(coco(), taxonomy))
    prepared = []
    for class_id in range(class_count):
        gt_counts = np.asarray([
            sum(item["class_id"] == class_id for item in gt[name]) for name in image_names
        ], dtype=np.int32)
        candidates = [row for row in predictions if row["class_id"] == class_id]
        candidates.sort(key=lambda row: row["confidence"], reverse=True)
        pred_images = np.asarray([image_index[row["file_name"]] for row in candidates], dtype=np.int32)
        thresholds = []
        for threshold in np.arange(.5, .96, .05):
            tp, fp, _, _ = class_at_iou(gt, predictions, class_id, float(threshold), False)
            thresholds.append((tp.astype(np.uint8), fp.astype(np.uint8)))
        prepared.append((gt_counts, pred_images, thresholds))
    return prepared, len(image_names)


def evaluate_cached(prepared, multiplicities: np.ndarray):
    class_metrics = []
    recall_points = np.linspace(0., 1., 101)
    for gt_counts, pred_images, thresholds in prepared:
        support = int(np.dot(gt_counts, multiplicities))
        if support == 0:
            continue
        weights = multiplicities[pred_images] if len(pred_images) else np.empty(0, dtype=np.int32)
        present = weights > 0
        repeat_counts = weights[present]
        aps = []
        best_f1 = 0.
        for threshold_index, (tp, fp) in enumerate(thresholds):
            expanded_tp = np.repeat(tp[present], repeat_counts)
            expanded_fp = np.repeat(fp[present], repeat_counts)
            if len(expanded_tp):
                cum_tp = np.cumsum(expanded_tp)
                cum_fp = np.cumsum(expanded_fp)
                recall = cum_tp / support
                precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
                ap = np.mean([precision[recall >= point].max()
                              if np.any(recall >= point) else 0. for point in recall_points])
                if threshold_index == 0:
                    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
                    best_f1 = float(f1.max())
            else:
                ap = 0.
            aps.append(float(ap))
        class_metrics.append((aps[0], float(np.mean(aps)), best_f1))
    values = np.asarray(class_metrics, dtype=float)
    return {"ap50": float(values[:, 0].mean()),
            "map50_95": float(values[:, 1].mean()),
            "f1": float(values[:, 2].mean())}


def validate_cache(prepared, original_arguments, image_count: int):
    rng = np.random.default_rng(91027)
    for _ in range(3):
        sample = rng.integers(0, image_count, size=image_count)
        multiplicities = np.bincount(sample, minlength=image_count).astype(np.int32)
        cached = evaluate_cached(prepared, multiplicities)
        original = analysis.evaluate_sample(*original_arguments, sample)
        for metric in METRICS:
            if not np.isclose(cached[metric], original[metric], rtol=0, atol=1e-12):
                raise AssertionError(f"Cached bootstrap differs for {metric}: {cached[metric]} vs {original[metric]}")


def bootstrap_seed(model_seed: int, iterations: int, resample_seed: int):
    loaded = {}
    prepared = {}
    image_count = None
    for label, (taxonomy, path) in configurations(model_seed).items():
        gt = analysis.ground_truth(taxonomy)
        arguments = (gt, load_json(path), len(taxonomy_names(coco(), taxonomy)))
        loaded[label] = arguments
        prepared[label], count = prepare_evaluator(taxonomy, path)
        image_count = count if image_count is None else image_count
        if count != image_count:
            raise AssertionError("Evaluation configurations use different image sets.")
    validate_cache(prepared["fine_direct"], loaded["fine_direct"], image_count)
    rng = np.random.default_rng(resample_seed)
    distributions = {label: {metric: [] for metric in METRICS} for label in prepared}
    for iteration in range(iterations):
        sample = rng.integers(0, image_count, size=image_count)
        multiplicities = np.bincount(sample, minlength=image_count).astype(np.int32)
        for label, cached in prepared.items():
            result = evaluate_cached(cached, multiplicities)
            for metric in METRICS:
                distributions[label][metric].append(result[metric])
        if (iteration + 1) % max(1, iterations // 10) == 0:
            print(f"seed {model_seed}: bootstrap {iteration + 1}/{iterations}", flush=True)
    rows = []
    for comparison, (left, right) in COMPARISONS.items():
        for metric in METRICS:
            difference = np.asarray(distributions[left][metric]) - np.asarray(distributions[right][metric])
            low, high = np.quantile(difference, [.025, .975])
            rows.append({"seed": model_seed, "comparison": comparison, "metric": metric,
                         "iterations": iterations, "bootstrap_mean_difference": float(difference.mean()),
                         "ci95_low": float(low), "ci95_high": float(high),
                         "includes_zero": bool(low <= 0 <= high),
                         "direction": "positive" if difference.mean() > 0 else "negative"})
    root = result_root(model_seed)
    save_csv(root / "bootstrap_comparisons.csv", rows, list(rows[0]))
    save_json(root / "bootstrap_config.json", {
        "model_seed": model_seed, "method": "paired nonparametric percentile bootstrap by test image",
        "implementation": "cached exact one-to-one matches; validated against original evaluator",
        "iterations": iterations, "confidence_level": .95, "random_seed": resample_seed,
        "unit": "image", "test_images": image_count,
    })
    return rows


def error_events(model_seed: int, confidence: float = .25, match_iou: float = .5):
    data = coco()
    fine_names = taxonomy_names(data, "fine")
    material_names = taxonomy_names(data, "material")
    mapping = analysis.fine_to_material()
    gt = analysis.ground_truth("fine")
    predictions = [row for row in load_json(result_root(model_seed) / "predictions/fine.json")
                   if row["confidence"] >= confidence]
    by_image = defaultdict(list)
    for row in predictions:
        by_image[row["file_name"]].append(row)
    events = []
    for file_name, truths in gt.items():
        unmatched = set(range(len(truths)))
        for pred in sorted(by_image[file_name], key=lambda row: row["confidence"], reverse=True):
            available = [(index, iou(pred["bbox_xyxy"], truths[index]["bbox_xyxy"])) for index in unmatched]
            truth_index, overlap = max(available, key=lambda item: item[1]) if available else (-1, 0.)
            if overlap >= match_iou:
                unmatched.remove(truth_index)
                true_id, pred_id = truths[truth_index]["class_id"], pred["class_id"]
                category = ("correct_fine" if true_id == pred_id else
                            "within_material_confusion" if mapping[true_id] == mapping[pred_id] else
                            "cross_material_confusion")
                events.append({"seed": model_seed, "file_name": file_name, "category": category,
                               "iou": overlap, "confidence": pred["confidence"],
                               "gt_box": truths[truth_index]["bbox_xyxy"], "pred_box": pred["bbox_xyxy"],
                               "true_class": fine_names[true_id], "predicted_class": fine_names[pred_id],
                               "true_material": material_names[mapping[true_id]],
                               "predicted_material": material_names[mapping[pred_id]]})
            else:
                all_overlaps = [iou(pred["bbox_xyxy"], truth["bbox_xyxy"]) for truth in truths]
                best_overlap = max(all_overlaps, default=0.)
                category = ("duplicate_detection" if best_overlap >= match_iou else
                            "localization_failure" if best_overlap >= .1 else
                            "background_false_positive")
                events.append({"seed": model_seed, "file_name": file_name, "category": category,
                               "iou": best_overlap, "confidence": pred["confidence"], "gt_box": None,
                               "pred_box": pred["bbox_xyxy"], "true_class": None,
                               "predicted_class": fine_names[pred["class_id"]], "true_material": None,
                               "predicted_material": material_names[mapping[pred["class_id"]]]})
        for truth_index in unmatched:
            true_id = truths[truth_index]["class_id"]
            events.append({"seed": model_seed, "file_name": file_name, "category": "missed_detection",
                           "iou": 0., "confidence": None, "gt_box": truths[truth_index]["bbox_xyxy"],
                           "pred_box": None, "true_class": fine_names[true_id], "predicted_class": None,
                           "true_material": material_names[mapping[true_id]], "predicted_material": None})
    save_json(result_root(model_seed) / "fine_error_events.json", events)
    return events


def aggregate_errors(all_events: dict[int, list[dict]]):
    categories = ("correct_fine", "within_material_confusion", "cross_material_confusion",
                  "localization_failure", "duplicate_detection", "background_false_positive",
                  "missed_detection")
    rows = []
    outcomes = defaultdict(dict)
    for seed, events in all_events.items():
        counts = Counter(row["category"] for row in events)
        spatial = sum(counts[name] for name in categories[:3])
        predictions = sum(counts[name] for name in categories[:-1])
        test_instances = spatial + counts["missed_detection"]
        for category in categories:
            denominator_name, denominator = (("spatial_matches", spatial) if category in categories[:3]
                                              else ("test_predictions", predictions) if category != "missed_detection"
                                              else ("test_instances", test_instances))
            rows.append({"seed": seed, "category": category, "count": counts[category],
                         "denominator": denominator_name, "rate": counts[category] / denominator})
        rows.append({"seed": seed, "category": "semantic_confusion_among_matches",
                     "count": counts["within_material_confusion"] + counts["cross_material_confusion"],
                     "denominator": "spatial_matches",
                     "rate": (counts["within_material_confusion"] + counts["cross_material_confusion"]) / spatial})
        for event in events:
            if event["gt_box"] is None:
                continue
            key = event["file_name"] + "|" + ",".join(f"{value:.3f}" for value in event["gt_box"])
            outcomes[key][seed] = event
    save_csv(ROOT / "results/multiseed_error_raw.csv", rows, list(rows[0]))
    frame = pd.DataFrame(rows)
    summary = []
    for category, group in frame.groupby("category", sort=False):
        summary.append({"category": category, "denominator": group.denominator.iloc[0],
                        "mean_count": group["count"].mean(), "std_count": group["count"].std(ddof=1),
                        "mean_rate": group.rate.mean(), "std_rate": group.rate.std(ddof=1),
                        "min_rate": group.rate.min(), "max_rate": group.rate.max()})
    save_csv(ROOT / "results/multiseed_error_summary.csv", summary, list(summary[0]))
    candidate_groups = {"stable_correct": [], "stable_semantic_error": [], "seed_disagreement": []}
    for key, seed_events in outcomes.items():
        if len(seed_events) != len(SEEDS):
            continue
        statuses = [seed_events[seed]["category"] for seed in SEEDS]
        item = {"key": key, "file_name": seed_events[42]["file_name"],
                "gt_box": seed_events[42]["gt_box"], "true_class": seed_events[42]["true_class"],
                "outcomes": {str(seed): {"category": seed_events[seed]["category"],
                                          "predicted_class": seed_events[seed]["predicted_class"],
                                          "confidence": seed_events[seed]["confidence"],
                                          "iou": seed_events[seed]["iou"]} for seed in SEEDS}}
        if all(status == "correct_fine" for status in statuses):
            candidate_groups["stable_correct"].append(item)
        elif all("confusion" in status for status in statuses):
            candidate_groups["stable_semantic_error"].append(item)
        elif len(set(statuses)) > 1:
            candidate_groups["seed_disagreement"].append(item)
    for items in candidate_groups.values():
        items.sort(key=lambda item: ((item["gt_box"][2]-item["gt_box"][0]) *
                                     (item["gt_box"][3]-item["gt_box"][1])), reverse=True)
        del items[30:]
    save_json(ROOT / "results/multiseed_qualitative_candidates.json", candidate_groups)


def aggregate_bootstrap(rows):
    save_csv(ROOT / "results/bootstrap_multiseed.csv", rows, list(rows[0]))
    frame = pd.DataFrame(rows)
    summary = []
    for (comparison, metric), group in frame.groupby(["comparison", "metric"], sort=False):
        directions = set(group.direction)
        excludes = ~group.includes_zero.astype(bool)
        summary.append({"comparison": comparison, "metric": metric,
                        "mean_difference_across_seeds": group.bootstrap_mean_difference.mean(),
                        "std_difference_across_seeds": group.bootstrap_mean_difference.std(ddof=1),
                        "seeds_excluding_zero": int(excludes.sum()), "num_seeds": len(group),
                        "consistent_direction": len(directions) == 1,
                        "all_exclude_zero_same_direction": bool(excludes.all() and len(directions) == 1),
                        "seed_intervals": "; ".join(
                            f"{int(row.seed)}:[{row.ci95_low:.4f},{row.ci95_high:.4f}]"
                            for row in group.itertuples())})
    save_csv(ROOT / "results/bootstrap_multiseed_summary.csv", summary, list(summary[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--resample-seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--force-bootstrap", action="store_true")
    args = parser.parse_args()
    bootstrap_rows = []
    all_events = {}
    for seed in args.seeds:
        path = result_root(seed) / "bootstrap_comparisons.csv"
        if path.exists() and not args.force_bootstrap:
            existing = pd.read_csv(path)
            if len(existing) == 9 and set(existing.iterations) == {args.iterations}:
                records = existing.to_dict("records")
                for row in records:
                    row["seed"] = seed
                    row["direction"] = "positive" if row["bootstrap_mean_difference"] > 0 else "negative"
                bootstrap_rows.extend(records)
            else:
                bootstrap_rows.extend(bootstrap_seed(seed, args.iterations, args.resample_seed))
        else:
            bootstrap_rows.extend(bootstrap_seed(seed, args.iterations, args.resample_seed))
        all_events[seed] = error_events(seed)
    aggregate_bootstrap(bootstrap_rows)
    aggregate_errors(all_events)
    print("Multiseed robustness evaluation completed.")


if __name__ == "__main__":
    main()

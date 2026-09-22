from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluation_utils import class_at_iou, evaluate_predictions, iou
from pipeline_utils import (
    ROOT,
    annotations_by_image,
    category_map,
    coco,
    dataset_dir,
    load_json,
    save_csv,
    save_json,
    split_data,
    taxonomy_names,
)


def fine_to_material() -> dict[int, int]:
    data = coco()
    fine = category_map(data, "fine")
    material = category_map(data, "material")
    return {fine[category_id]: material[category_id] for category_id in fine}


def transform_predictions(target: str) -> Path:
    source = load_json(ROOT / "results" / "predictions" / "fine.json")
    mapping = fine_to_material()
    if target == "fine":
        converted = source
    elif target == "material":
        converted = [{**row, "class_id": mapping[row["class_id"]]} for row in source]
    elif target == "binary":
        converted = [{**row, "class_id": 0} for row in source]
    else:
        raise ValueError(target)
    path = ROOT / "results" / "predictions" / f"fine_as_{target}.json"
    save_json(path, converted)
    return path


def hierarchical_evaluation() -> list[dict[str, Any]]:
    rows = []
    for target in ("fine", "material", "binary"):
        summary, _ = evaluate_predictions(target, transform_predictions(target), agnostic=False)
        rows.append({"source_model": "fine", "evaluation_taxonomy": target, **summary})
    save_csv(ROOT / "results" / "fine_hierarchical_evaluation.csv", rows, list(rows[0]))
    return rows


def ground_truth(taxonomy: str) -> dict[str, list[dict[str, Any]]]:
    data = coco()
    test = set(split_data()["test"])
    images = {row["id"]: row for row in data["images"] if Path(row["file_name"]).as_posix() in test}
    grouped = annotations_by_image(data)
    remap = category_map(data, taxonomy)
    result: dict[str, list[dict[str, Any]]] = {}
    for image in images.values():
        name = Path(image["file_name"]).as_posix()
        with Image.open(dataset_dir(taxonomy) / "images" / "test" / Path(name)) as actual:
            sx, sy = actual.width / image["width"], actual.height / image["height"]
        result[name] = []
        for ann in grouped[image["id"]]:
            x, y, w, h = map(float, ann["bbox"])
            box = [max(0.0, x) * sx, max(0.0, y) * sy,
                   min(float(image["width"]), x + w) * sx,
                   min(float(image["height"]), y + h) * sy]
            result[name].append({"class_id": remap[ann["category_id"]], "bbox_xyxy": box})
    return result


def evaluate_sample(
    gt: dict[str, list[dict[str, Any]]],
    predictions: list[dict[str, Any]],
    num_classes: int,
    sample: np.ndarray,
) -> dict[str, float]:
    names = list(gt)
    pred_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        pred_by_image[pred["file_name"]].append(pred)
    sampled_gt: dict[str, list[dict[str, Any]]] = {}
    sampled_predictions = []
    occurrences: Counter[str] = Counter()
    for image_index in sample:
        name = names[int(image_index)]
        copy_index = occurrences[name]
        occurrences[name] += 1
        synthetic = f"{copy_index}:{name}"
        sampled_gt[synthetic] = gt[name]
        sampled_predictions.extend({**pred, "file_name": synthetic} for pred in pred_by_image.get(name, []))
    thresholds = np.arange(0.5, 0.96, 0.05)
    per_class = []
    for class_id in range(num_classes):
        aps = []
        best_f1 = 0.0
        support = 0
        for threshold in thresholds:
            tp, fp, _, support = class_at_iou(sampled_gt, sampled_predictions, class_id, float(threshold), False)
            cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
            recall = cum_tp / max(support, 1)
            precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
            if support:
                points = np.linspace(0, 1, 101)
                aps.append(float(np.mean([precision[recall >= point].max() if np.any(recall >= point) else 0.0 for point in points])))
            else:
                aps.append(float("nan"))
            if threshold == 0.5 and len(precision):
                f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
                best_f1 = float(f1.max())
        if support:
            per_class.append((aps[0], float(np.nanmean(aps)), best_f1))
    values = np.asarray(per_class, dtype=float)
    return {"ap50": float(values[:, 0].mean()), "map50_95": float(values[:, 1].mean()), "f1": float(values[:, 2].mean())}


def bootstrap(iterations: int, seed: int) -> list[dict[str, Any]]:
    data = coco()
    configurations = {
        "fine_direct": ("fine", ROOT / "results/predictions/fine.json"),
        "material_direct": ("material", ROOT / "results/predictions/material.json"),
        "fine_as_material": ("material", ROOT / "results/predictions/fine_as_material.json"),
        "fine_as_binary": ("binary", ROOT / "results/predictions/fine_as_binary.json"),
    }
    loaded = {}
    for label, (taxonomy, path) in configurations.items():
        loaded[label] = (
            ground_truth(taxonomy),
            load_json(path),
            len(taxonomy_names(data, taxonomy)),
        )
    image_count = len(next(iter(loaded.values()))[0])
    rng = np.random.default_rng(seed)
    distributions = {label: {metric: [] for metric in ("ap50", "map50_95", "f1")} for label in loaded}
    for iteration in range(iterations):
        sample = rng.integers(0, image_count, size=image_count)
        for label, arguments in loaded.items():
            result = evaluate_sample(*arguments, sample)
            for metric, value in result.items():
                distributions[label][metric].append(value)
        if (iteration + 1) % max(1, iterations // 10) == 0:
            print(f"Bootstrap {iteration + 1}/{iterations}", flush=True)
    comparisons = {
        "fine_direct_minus_material_direct": ("fine_direct", "material_direct"),
        "fine_as_material_minus_fine_direct": ("fine_as_material", "fine_direct"),
        "fine_as_binary_minus_fine_direct": ("fine_as_binary", "fine_direct"),
    }
    rows = []
    for comparison, (left, right) in comparisons.items():
        for metric in ("ap50", "map50_95", "f1"):
            differences = np.asarray(distributions[left][metric]) - np.asarray(distributions[right][metric])
            rows.append({
                "comparison": comparison,
                "metric": metric,
                "iterations": iterations,
                "bootstrap_mean_difference": float(differences.mean()),
                "ci95_low": float(np.quantile(differences, 0.025)),
                "ci95_high": float(np.quantile(differences, 0.975)),
                "includes_zero": bool(np.quantile(differences, 0.025) <= 0 <= np.quantile(differences, 0.975)),
            })
    save_csv(ROOT / "results/bootstrap_comparisons.csv", rows, list(rows[0]))
    save_json(ROOT / "results/bootstrap_config.json", {
        "method": "paired nonparametric percentile bootstrap by test image",
        "iterations": iterations,
        "confidence_level": 0.95,
        "random_seed": seed,
        "unit": "image",
        "note": "All comparisons use the controlled evaluator and identical resampled image indices.",
    })
    return rows


def matched_error_events(confidence: float = 0.25, match_iou: float = 0.5) -> list[dict[str, Any]]:
    data = coco()
    fine_names = taxonomy_names(data, "fine")
    material_names = taxonomy_names(data, "material")
    mapping = fine_to_material()
    gt = ground_truth("fine")
    predictions = [row for row in load_json(ROOT / "results/predictions/fine.json") if row["confidence"] >= confidence]
    pred_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        pred_by_image[pred["file_name"]].append(pred)
    events = []
    for file_name, truths in gt.items():
        unmatched = set(range(len(truths)))
        for pred in sorted(pred_by_image.get(file_name, []), key=lambda row: row["confidence"], reverse=True):
            available = [(index, iou(pred["bbox_xyxy"], truths[index]["bbox_xyxy"])) for index in unmatched]
            if available:
                truth_index, overlap = max(available, key=lambda item: item[1])
            else:
                truth_index, overlap = -1, 0.0
            if overlap >= match_iou:
                unmatched.remove(truth_index)
                true_id, pred_id = truths[truth_index]["class_id"], pred["class_id"]
                if true_id == pred_id:
                    category = "correct_fine"
                elif mapping[true_id] == mapping[pred_id]:
                    category = "within_material_confusion"
                else:
                    category = "cross_material_confusion"
                events.append({"file_name": file_name, "category": category, "iou": overlap,
                               "confidence": pred["confidence"], "gt_box": truths[truth_index]["bbox_xyxy"],
                               "pred_box": pred["bbox_xyxy"], "true_class": fine_names[true_id],
                               "predicted_class": fine_names[pred_id], "true_material": material_names[mapping[true_id]],
                               "predicted_material": material_names[mapping[pred_id]]})
            else:
                all_overlaps = [iou(pred["bbox_xyxy"], truth["bbox_xyxy"]) for truth in truths]
                best_overlap = max(all_overlaps, default=0.0)
                if best_overlap >= match_iou:
                    category = "duplicate_detection"
                elif best_overlap >= 0.1:
                    category = "localization_failure"
                else:
                    category = "background_false_positive"
                events.append({"file_name": file_name, "category": category, "iou": best_overlap,
                               "confidence": pred["confidence"], "gt_box": None, "pred_box": pred["bbox_xyxy"],
                               "true_class": None, "predicted_class": fine_names[pred["class_id"]],
                               "true_material": None, "predicted_material": material_names[mapping[pred["class_id"]]]})
        for truth_index in unmatched:
            true_id = truths[truth_index]["class_id"]
            events.append({"file_name": file_name, "category": "missed_detection", "iou": 0.0,
                           "confidence": None, "gt_box": truths[truth_index]["bbox_xyxy"], "pred_box": None,
                           "true_class": fine_names[true_id], "predicted_class": None,
                           "true_material": material_names[mapping[true_id]], "predicted_material": None})
    return events


def error_analysis() -> list[dict[str, Any]]:
    events = matched_error_events()
    save_json(ROOT / "results/fine_error_events.json", events)
    counts = Counter(event["category"] for event in events)
    rows = [{"category": category, "count": counts[category]} for category in (
        "correct_fine", "within_material_confusion", "cross_material_confusion",
        "localization_failure", "duplicate_detection", "background_false_positive", "missed_detection")]
    save_csv(ROOT / "results/fine_error_decomposition.csv", rows, list(rows[0]))
    confusions = Counter((event["true_class"], event["predicted_class"], event["category"])
                         for event in events if "confusion" in event["category"])
    confusion_rows = [{"true_class": key[0], "predicted_class": key[1], "error_scope": key[2], "count": count}
                      for key, count in confusions.most_common()]
    save_csv(ROOT / "results/fine_confusion_pairs.csv", confusion_rows,
             ["true_class", "predicted_class", "error_scope", "count"])
    return events


def draw_example(event: dict[str, Any], target: Path) -> None:
    source = dataset_dir("fine") / "images/test" / Path(event["file_name"])
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    if event["gt_box"]:
        draw.rectangle(event["gt_box"], outline=(0, 220, 70), width=4)
        draw.text((event["gt_box"][0], max(0, event["gt_box"][1] - 14)), f"GT: {event['true_class']}", fill=(0, 255, 80), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    if event["pred_box"]:
        draw.rectangle(event["pred_box"], outline=(255, 45, 45), width=4)
        draw.text((event["pred_box"][0], event["pred_box"][1] + 3), f"Pred: {event['predicted_class']}", fill=(255, 70, 70), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    target.parent.mkdir(parents=True, exist_ok=True)
    image.thumbnail((1200, 900))
    image.save(target, quality=92)


def qualitative_examples(events: list[dict[str, Any]]) -> None:
    categories = ["correct_fine", "within_material_confusion", "cross_material_confusion", "localization_failure", "missed_detection"]
    selected = []
    output = ROOT / "figures/error_examples"
    for category in categories:
        options = [event for event in events if event["category"] == category]
        if not options:
            continue
        if category == "missed_detection":
            event = options[0]
        else:
            event = max(options, key=lambda row: row["confidence"] or 0.0)
        path = output / f"{category}.jpg"
        draw_example(event, path)
        selected.append((category, path, event))
    if not selected:
        return
    fig, axes = plt.subplots(1, len(selected), figsize=(4.2 * len(selected), 4.2))
    axes = np.atleast_1d(axes)
    for axis, (category, path, event) in zip(axes, selected):
        axis.imshow(Image.open(path))
        axis.set_title(category.replace("_", " "))
        axis.axis("off")
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures/05_error_examples.png", dpi=300, bbox_inches="tight")
    fig.savefig(ROOT / "figures/05_error_examples.pdf", bbox_inches="tight")
    plt.close(fig)
    save_json(ROOT / "results/qualitative_examples.json", [
        {"category": category, "image": event["file_name"], "rendered_file": str(path.relative_to(ROOT)).replace("\\", "/")}
        for category, path, event in selected])


def main() -> None:
    parser = argparse.ArgumentParser(description="Hierarchical, bootstrap, semantic-error and qualitative analyses.")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    hierarchical_evaluation()
    bootstrap(args.bootstrap_iterations, args.bootstrap_seed)
    events = error_analysis()
    qualitative_examples(events)
    print("Strengthened analyses completed.")


if __name__ == "__main__":
    main()

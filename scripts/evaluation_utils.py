from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline_utils import ROOT, annotations_by_image, category_map, coco, dataset_dir, load_json, split_data, taxonomy_names


def iou(box: list[float], other: list[float]) -> float:
    x1 = max(box[0], other[0]); y1 = max(box[1], other[1])
    x2 = min(box[2], other[2]); y2 = min(box[3], other[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = max(0.0, other[2] - other[0]) * max(0.0, other[3] - other[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    recall_points = np.linspace(0, 1, 101)
    values = [precision[recall >= point].max() if np.any(recall >= point) else 0.0 for point in recall_points]
    return float(np.mean(values))


def class_at_iou(gt_by_image: dict[str, list[dict]], predictions: list[dict], class_id: int, threshold: float, agnostic: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    gt = {name: [item for item in items if agnostic or item["class_id"] == class_id] for name, items in gt_by_image.items()}
    candidates = [p for p in predictions if agnostic or p["class_id"] == class_id]
    candidates.sort(key=lambda p: p["confidence"], reverse=True)
    matched: dict[str, set[int]] = defaultdict(set)
    tp = np.zeros(len(candidates)); fp = np.zeros(len(candidates)); scores = np.array([p["confidence"] for p in candidates])
    for index, prediction in enumerate(candidates):
        options = gt.get(prediction["file_name"], [])
        overlaps = [iou(prediction["bbox_xyxy"], item["bbox_xyxy"]) if j not in matched[prediction["file_name"]] else -1 for j, item in enumerate(options)]
        best = int(np.argmax(overlaps)) if overlaps else -1
        if best >= 0 and overlaps[best] >= threshold:
            tp[index] = 1; matched[prediction["file_name"]].add(best)
        else:
            fp[index] = 1
    return tp, fp, scores, sum(len(items) for items in gt.values())


def evaluate_predictions(taxonomy: str, predictions_path: Path, agnostic: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = coco(); split = split_data(); test_names = set(split["test"])
    images = {i["id"]: i for i in data["images"] if Path(i["file_name"]).as_posix() in test_names}
    remap = category_map(data, taxonomy); grouped = annotations_by_image(data)
    gt_by_image: dict[str, list[dict]] = {}
    for image in images.values():
        name = Path(image["file_name"]).as_posix()
        with Image.open(dataset_dir(taxonomy) / "images" / "test" / Path(name)) as actual:
            scale_x = actual.width / image["width"]
            scale_y = actual.height / image["height"]
        gt_by_image[name] = []
        for ann in grouped[image["id"]]:
            x, y, w, h = map(float, ann["bbox"])
            x1 = max(0.0, x); y1 = max(0.0, y)
            x2 = min(float(image["width"]), x + w); y2 = min(float(image["height"]), y + h)
            gt_by_image[name].append({"class_id": 0 if agnostic else remap[ann["category_id"]], "bbox_xyxy": [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]})
    predictions = load_json(predictions_path)
    if agnostic:
        predictions = [{**p, "class_id": 0} for p in predictions]
    class_ids = [0] if agnostic else list(range(len(taxonomy_names(data, taxonomy))))
    thresholds = np.arange(0.5, 0.96, 0.05)
    rows = []
    for class_id in class_ids:
        aps = []; best_p = best_r = best_f1 = 0.0; support = 0
        for threshold in thresholds:
            tp, fp, scores, n_gt = class_at_iou(gt_by_image, predictions, class_id, float(threshold), agnostic)
            support = n_gt
            cum_tp = np.cumsum(tp); cum_fp = np.cumsum(fp)
            recall = cum_tp / max(n_gt, 1); precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
            aps.append(average_precision(recall, precision) if n_gt else float("nan"))
            if abs(threshold - 0.5) < 1e-9 and len(precision):
                f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
                best = int(np.argmax(f1)); best_p = float(precision[best]); best_r = float(recall[best]); best_f1 = float(f1[best])
        rows.append({
            "class_id": class_id,
            "class_name": "Litter" if agnostic else taxonomy_names(data, taxonomy)[class_id],
            "test_instances": support,
            "precision": best_p, "recall": best_r, "f1": best_f1,
            "ap50": aps[0], "map50_95": float(np.nanmean(aps)) if support else float("nan"),
        })
    valid = [row for row in rows if row["test_instances"] > 0]
    keys = ("precision", "recall", "f1", "ap50", "map50_95")
    summary = {key: float(np.mean([row[key] for row in valid])) if valid else 0.0 for key in keys}
    summary["map50"] = summary["ap50"]
    summary.update({"taxonomy": taxonomy, "evaluation": "class_agnostic" if agnostic else "class_aware", "num_classes_evaluated": len(valid), "test_images": len(test_names), "test_instances": sum(len(x) for x in gt_by_image.values())})
    return summary, rows


def semantic_confusions(predictions_path: Path, confidence: float = 0.25, threshold: float = 0.5) -> list[dict[str, Any]]:
    data = coco(); names = taxonomy_names(data, "fine"); remap = category_map(data, "fine")
    split = split_data(); test_names = set(split["test"]); by_image = annotations_by_image(data)
    images = {i["id"]: i for i in data["images"] if Path(i["file_name"]).as_posix() in test_names}
    gts: dict[str, list[dict]] = {}
    for image in images.values():
        key = Path(image["file_name"]).as_posix(); gts[key] = []
        with Image.open(dataset_dir("fine") / "images" / "test" / Path(key)) as actual:
            scale_x = actual.width / image["width"]; scale_y = actual.height / image["height"]
        for ann in by_image[image["id"]]:
            x, y, w, h = map(float, ann["bbox"])
            x1 = max(0.0, x); y1 = max(0.0, y); x2 = min(float(image["width"]), x+w); y2 = min(float(image["height"]), y+h)
            gts[key].append({"class_id": remap[ann["category_id"]], "bbox": [x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y]})
    preds = [p for p in load_json(predictions_path) if p["confidence"] >= confidence]
    pred_by_image: dict[str, list[dict]] = defaultdict(list)
    for pred in preds: pred_by_image[pred["file_name"]].append(pred)
    counts = Counter()
    for file_name, items in pred_by_image.items():
        unmatched = set(range(len(gts.get(file_name, []))))
        for pred in sorted(items, key=lambda p: p["confidence"], reverse=True):
            overlaps = [(j, iou(pred["bbox_xyxy"], gts[file_name][j]["bbox"])) for j in unmatched]
            if not overlaps: continue
            match, score = max(overlaps, key=lambda item: item[1])
            if score >= threshold:
                unmatched.remove(match); true = gts[file_name][match]["class_id"]
                if true != pred["class_id"]:
                    counts[(true, pred["class_id"])] += 1
    return [{"ground_truth": names[t], "predicted": names[p], "count": count} for (t, p), count in counts.most_common(20)]

"""Audit quantitative interpretations and preview localization examples."""
import importlib
import json
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import pandas as pd

from pipeline_utils import ROOT, load_json, save_json
from evaluation_utils import iou


def taxonomy_events(analysis, taxonomy):
    truths = analysis.ground_truth(taxonomy)
    predictions = [p for p in load_json(ROOT / f"results/predictions/{taxonomy}.json")
                   if p["confidence"] >= .25]
    by_image = defaultdict(list)
    for pred in predictions:
        by_image[pred["file_name"]].append(pred)
    names = analysis.taxonomy_names(analysis.coco(), taxonomy)
    events = []
    for file_name, image_truths in truths.items():
        unmatched = set(range(len(image_truths)))
        for pred in sorted(by_image[file_name], key=lambda p: p["confidence"], reverse=True):
            options = [(j, iou(pred["bbox_xyxy"], image_truths[j]["bbox_xyxy"])) for j in unmatched]
            j, overlap = max(options, key=lambda x: x[1]) if options else (-1, 0.)
            if overlap >= .5:
                unmatched.remove(j)
                truth = image_truths[j]
                category = "correct" if truth["class_id"] == pred["class_id"] else "semantic_error"
                events.append({"file_name": file_name, "category": category, "iou": overlap,
                               "confidence": pred["confidence"], "gt_box": truth["bbox_xyxy"],
                               "pred_box": pred["bbox_xyxy"], "true_class": names[truth["class_id"]],
                               "predicted_class": names[pred["class_id"]]})
        for j in unmatched:
            truth = image_truths[j]
            events.append({"file_name": file_name, "category": "missed", "iou": 0., "confidence": None,
                           "gt_box": truth["bbox_xyxy"], "pred_box": None,
                           "true_class": names[truth["class_id"]], "predicted_class": None})
    return events


def main():
    assets = importlib.import_module("17_generate_paper_assets")
    analysis = importlib.import_module("14_strengthen_analysis")
    events = load_json(ROOT / "results/fine_error_events.json")
    truths = analysis.ground_truth("fine")
    names = analysis.taxonomy_names(analysis.coco(), "fine")
    candidates = []
    for event_index, event in enumerate(events):
        if event["category"] != "localization_failure":
            continue
        gt = max(truths[event["file_name"]], key=lambda g: iou(g["bbox_xyxy"], event["pred_box"]))
        candidates.append({**event, "event_index": event_index,
                           "gt_box": gt["bbox_xyxy"], "true_class": names[gt["class_id"]]})
    out = ROOT / "paper/tmp/review"
    out.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(candidates), 12):
        fig, axes = plt.subplots(3, 4, figsize=(14, 11))
        for ax in axes.flat:
            ax.axis("off")
        for ax, event in zip(axes.flat, candidates[start:start+12]):
            title = (f"#{event['event_index']} IoU={event['iou']:.3f}\n"
                     f"{event['true_class']} -> {event['predicted_class']}")
            assets._draw_event(ax, event, title)
        fig.tight_layout()
        fig.savefig(out / f"candidates_{start:02d}.png", dpi=120)
        plt.close(fig)
    missed = [e for e in events if e["category"] == "missed_detection" and e.get("gt_box")]
    missed.sort(key=lambda e: (e["gt_box"][2]-e["gt_box"][0]) * (e["gt_box"][3]-e["gt_box"][1]), reverse=True)
    fig, axes = plt.subplots(3, 4, figsize=(14, 11))
    for ax, event in zip(axes.flat, missed[:12]):
        area = (event["gt_box"][2]-event["gt_box"][0]) * (event["gt_box"][3]-event["gt_box"][1])
        assets._draw_event(ax, event, f"{event['true_class']}\narea={area:.0f}", show_pred=False)
    fig.tight_layout()
    fig.savefig(out / "missed_candidates.png", dpi=120)
    plt.close(fig)
    for taxonomy in ("material", "binary"):
        taxonomy_rows = taxonomy_events(analysis, taxonomy)
        for category in ("correct", "semantic_error", "missed"):
            options = [e for e in taxonomy_rows if e["category"] == category]
            options.sort(key=lambda e: ((e["gt_box"][2]-e["gt_box"][0]) *
                                        (e["gt_box"][3]-e["gt_box"][1])) *
                                       (e.get("confidence") or 1), reverse=True)
            if not options:
                continue
            fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
            for ax in axes.flat:
                ax.axis("off")
            for ax, event in zip(axes.flat, options[:8]):
                label = event["true_class"]
                if event.get("predicted_class") and event["predicted_class"] != label:
                    label += f" -> {event['predicted_class']}"
                assets._draw_event(ax, event, label, show_pred=event.get("pred_box") is not None)
            fig.tight_layout()
            fig.savefig(out / f"{taxonomy}_{category}.png", dpi=120)
            plt.close(fig)
    counts = Counter(event["category"] for event in events)
    loc = [event for event in events if event["category"] == "localization_failure"]
    classes = pd.read_csv(ROOT / "results/fine_class_metrics.csv")
    distribution = pd.read_csv(ROOT / "results/class_distribution.csv")
    merged = distribution.merge(classes, on="class_id", suffixes=("_distribution", "_metrics"))
    report = {
        "error_counts": dict(counts),
        "localization_events_with_iou_at_least_05": sum(e["iou"] >= .5 for e in loc),
        "matched_instances": sum(counts[k] for k in ("correct_fine", "within_material_confusion", "cross_material_confusion")),
        "semantic_confusions_among_matches": counts["within_material_confusion"] + counts["cross_material_confusion"],
        "fine_classes_with_test_support": int((classes.test_instances > 0).sum()),
        "frequency_groups": {
            str(group): {"total_classes": len(frame), "classes_with_test_support": int((frame.test_instances_metrics > 0).sum())}
            for group, frame in merged.groupby("frequency_group")
        },
        "bootstrap_difference_definition": "Mean of bootstrap differences, not observed point difference",
        "qualitative_candidates": candidates,
    }
    save_json(ROOT / "paper/review_audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "qualitative_candidates"}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from pipeline_utils import ROOT, annotations_by_image, category_map, coco, config, save_csv, save_json, split_data, split_lookup


def main() -> None:
    data = coco(); split = split_data(); membership = split_lookup(split)
    images = {i["id"]: Path(i["file_name"]).as_posix() for i in data["images"]}
    names = {c["id"]: c["name"] for c in data["categories"]}; remap = category_map(data, "fine")
    counts = {part: Counter() for part in ("train", "val", "test")}
    for ann in data["annotations"]:
        counts[membership[images[ann["image_id"]]]][remap[ann["category_id"]]] += 1
    thresholds = config()["frequency_groups"]
    class_rows = []
    for original_id, class_name in names.items():
        class_id = remap[original_id]; train_count = counts["train"][class_id]
        group = "Rare" if train_count <= thresholds["rare_max"] else "Medium" if train_count <= thresholds["medium_max"] else "Frequent"
        class_rows.append({"class_id": class_id, "class_name": class_name, "train_instances": train_count, "val_instances": counts["val"][class_id], "test_instances": counts["test"][class_id], "frequency_group": group})
    save_csv(ROOT / "results" / "class_distribution.csv", class_rows, list(class_rows[0]))
    metrics_path = ROOT / "results" / "fine_class_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Run scripts/06_evaluate.py before long-tail analysis")
    frame = pd.DataFrame(class_rows).merge(pd.read_csv(metrics_path), on=["class_id", "class_name"], suffixes=("", "_metric"))
    groups = []
    for group in ("Rare", "Medium", "Frequent"):
        current = frame[frame["frequency_group"] == group]
        groups.append({
            "frequency_group": group, "num_classes": int(len(current)),
            "mean_train_instances": float(current["train_instances"].mean()),
            "mean_ap50": float(current["ap50"].mean()),
            "mean_ap50_95": float(current["map50_95"].mean()),
            "mean_recall": float(current["recall"].mean()),
        })
    save_csv(ROOT / "results" / "long_tail_groups.csv", groups, list(groups[0]))
    save_json(ROOT / "results" / "long_tail_config.json", {"thresholds": thresholds, "basis": "training instances only", "definitions": {"Rare": f"<= {thresholds['rare_max']}", "Medium": f"{thresholds['rare_max'] + 1}..{thresholds['medium_max']}", "Frequent": f"> {thresholds['medium_max']}"}})
    print(*groups, sep="\n")


if __name__ == "__main__":
    main()

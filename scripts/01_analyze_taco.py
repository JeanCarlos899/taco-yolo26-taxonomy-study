from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt

from pipeline_utils import ROOT, annotations_by_image, coco, image_path, save_csv, save_json


def main() -> None:
    data = coco()
    by_image = annotations_by_image(data)
    categories = {c["id"]: c["name"] for c in data["categories"]}
    counts = Counter(a["category_id"] for a in data["annotations"])
    invalid = []
    for annotation in data["annotations"]:
        image = next(i for i in data["images"] if i["id"] == annotation["image_id"])
        x, y, width, height = annotation["bbox"]
        reasons = []
        if width <= 0 or height <= 0:
            reasons.append("non_positive_size")
        if x < 0 or y < 0 or x + width > image["width"] or y + height > image["height"]:
            reasons.append("outside_image")
        if reasons:
            invalid.append({"annotation_id": annotation["id"], "image_id": image["id"], "file_name": image["file_name"], "reasons": reasons, "bbox": annotation["bbox"]})
    missing = [i["file_name"] for i in data["images"] if not image_path(i).exists()]
    rows = [{"category_id": cid, "category": categories[cid], "instances": counts[cid]} for cid in categories]
    rows.sort(key=lambda row: (-row["instances"], row["category"]))
    stats = {
        "num_images": len(data["images"]),
        "num_objects": len(data["annotations"]),
        "num_categories": len(data["categories"]),
        "images_without_objects": sum(not by_image[i["id"]] for i in data["images"]),
        "invalid_bounding_boxes": len(invalid),
        "missing_image_files": len(missing),
    }
    save_json(ROOT / "results" / "dataset_statistics.json", {**stats, "missing_images": missing, "invalid_boxes": invalid})
    save_csv(ROOT / "results" / "category_distribution.csv", rows, ["category_id", "category", "instances"])

    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    labels = [row["category"] for row in reversed(rows)]
    values = [row["instances"] for row in reversed(rows)]
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.barh(labels, values, color="#2878B5")
    ax.set_xscale("log")
    ax.set_xlabel("Annotated instances (log scale)")
    ax.set_ylabel("Original TACO category")
    ax.set_title("TACO class distribution")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"01_class_distribution.{suffix}", dpi=300, bbox_inches="tight")
    print(stats)


if __name__ == "__main__":
    main()

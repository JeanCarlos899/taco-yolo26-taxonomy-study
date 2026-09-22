from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

from pipeline_utils import (ROOT, TAXONOMIES, annotations_by_image, category_map, coco,
                            dataset_dir, hardlink_or_copy, image_path, load_json,
                            save_json, split_data, split_lookup, taxonomy_names)


def validate_material_mapping(data: dict) -> None:
    document = load_json(ROOT / "configs" / "material_mapping.json")
    official = {c["name"] for c in data["categories"]}
    mapped = set(document["mapping"])
    if official != mapped:
        raise ValueError(f"Material mapping mismatch. Missing={sorted(official-mapped)}, extra={sorted(mapped-official)}")
    invalid = {item["material"] for item in document["mapping"].values()} - set(document["macroclasses"])
    if invalid:
        raise ValueError(f"Unknown macroclasses: {sorted(invalid)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all YOLO taxonomies using the fixed image split.")
    parser.add_argument("--taxonomy", choices=(*TAXONOMIES, "all"), default="all")
    args = parser.parse_args()
    data = coco()
    validate_material_mapping(data)
    split = split_data()
    membership = split_lookup(split)
    by_image = annotations_by_image(data)
    selected = TAXONOMIES if args.taxonomy == "all" else (args.taxonomy,)
    missing = [i["file_name"] for i in data["images"] if not image_path(i).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images are missing. Run scripts/00_download_images.py first.")
    corrections = []
    for taxonomy in selected:
        out = dataset_dir(taxonomy)
        names = taxonomy_names(data, taxonomy)
        remap = category_map(data, taxonomy)
        link_modes = Counter()
        for image in data["images"]:
            key = Path(image["file_name"]).as_posix()
            part = membership[key]
            image_target = out / "images" / part / Path(image["file_name"])
            link_modes[hardlink_or_copy(image_path(image), image_target)] += 1
            label_target = out / "labels" / part / Path(image["file_name"]).with_suffix(".txt")
            label_target.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for ann in by_image[image["id"]]:
                x, y, width, height = map(float, ann["bbox"])
                if width <= 0 or height <= 0:
                    raise ValueError(f"Non-positive bbox in annotation {ann['id']}: {ann['bbox']}")
                x1 = max(0.0, x); y1 = max(0.0, y)
                x2 = min(float(image["width"]), x + width); y2 = min(float(image["height"]), y + height)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"BBox has no intersection with image in annotation {ann['id']}: {ann['bbox']}")
                is_outside = x < 0 or y < 0 or x + width > image["width"] or y + height > image["height"]
                if is_outside and taxonomy == selected[0]:
                    corrections.append({"annotation_id": ann["id"], "file_name": image["file_name"], "original_bbox": ann["bbox"], "clipped_bbox": [x1, y1, x2 - x1, y2 - y1]})
                x, y, width, height = x1, y1, x2 - x1, y2 - y1
                xc = (x + width / 2) / image["width"]
                yc = (y + height / 2) / image["height"]
                wn = width / image["width"]
                hn = height / image["height"]
                lines.append(f"{remap[ann['category_id']]} {xc:.8f} {yc:.8f} {wn:.8f} {hn:.8f}")
            label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        descriptor = {
            "path": str(out.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: name for i, name in enumerate(names)},
        }
        with (out / "dataset.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(descriptor, handle, sort_keys=False, allow_unicode=True)
        print(taxonomy, len(names), dict(link_modes))
    save_json(ROOT / "results" / "bbox_corrections.json", {"policy": "clip COCO boxes to image boundaries; official annotations.json remains unchanged", "count": len(corrections), "corrections": corrections})


if __name__ == "__main__":
    main()

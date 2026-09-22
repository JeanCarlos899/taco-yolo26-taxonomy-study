from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pipeline_utils import ROOT, TAXONOMIES, coco, dataset_dir, find_images, image_path, split_data, taxonomy_names


def validate_taxonomy(taxonomy: str, expected: dict[str, list[str]]) -> dict:
    root = dataset_dir(taxonomy)
    names = taxonomy_names(coco(), taxonomy)
    errors: list[str] = []
    counts = Counter()
    for part in ("train", "val", "test"):
        actual = []
        for file_name in expected[part]:
            image = root / "images" / part / Path(file_name)
            label = root / "labels" / part / Path(file_name).with_suffix(".txt")
            if not image.exists():
                errors.append(f"missing image: {image}")
            else:
                actual.append(Path(file_name).as_posix())
            if not label.exists():
                errors.append(f"missing label: {label}")
                continue
            for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                if len(fields) != 5:
                    errors.append(f"{label}:{line_no}: expected 5 fields")
                    continue
                try:
                    class_id = int(fields[0]); values = list(map(float, fields[1:]))
                except ValueError:
                    errors.append(f"{label}:{line_no}: non-numeric value")
                    continue
                if not 0 <= class_id < len(names):
                    errors.append(f"{label}:{line_no}: class id {class_id} out of range")
                if any(not 0 <= value <= 1 for value in values):
                    errors.append(f"{label}:{line_no}: normalized coordinate outside [0,1]")
                if values[2] <= 0 or values[3] <= 0:
                    errors.append(f"{label}:{line_no}: non-positive width/height")
                if values[0] - values[2] / 2 < -1e-6 or values[0] + values[2] / 2 > 1 + 1e-6 or values[1] - values[3] / 2 < -1e-6 or values[1] + values[3] / 2 > 1 + 1e-6:
                    errors.append(f"{label}:{line_no}: box extends outside image")
                counts[class_id] += 1
        if sorted(actual) != sorted(expected[part]):
            errors.append(f"{part}: image membership differs from split.json")
    if errors:
        raise RuntimeError(f"{taxonomy}: {len(errors)} validation errors\n" + "\n".join(errors[:30]))
    return {"taxonomy": taxonomy, "classes": len(names), "objects": sum(counts.values()), "class_counts": dict(sorted(counts.items()))}


def visualize(taxonomy: str, count: int, seed: int) -> None:
    root = dataset_dir(taxonomy)
    names = taxonomy_names(coco(), taxonomy)
    candidates = find_images(root / "images" / "train")
    chosen = random.Random(seed).sample(candidates, min(count, len(candidates)))
    preview = ROOT / "figures" / "label_previews" / taxonomy
    preview.mkdir(parents=True, exist_ok=True)
    for source in chosen:
        relative = source.relative_to(root / "images" / "train")
        label = root / "labels" / "train" / relative.with_suffix(".txt")
        with Image.open(source) as raw:
            canvas = raw.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for line in label.read_text(encoding="utf-8").splitlines():
            class_id, xc, yc, width, height = map(float, line.split())
            x1 = (xc - width / 2) * canvas.width; y1 = (yc - height / 2) * canvas.height
            x2 = (xc + width / 2) * canvas.width; y2 = (yc + height / 2) * canvas.height
            draw.rectangle((x1, y1, x2, y2), outline="#ff3030", width=max(2, canvas.width // 300))
            draw.text((x1 + 2, y1 + 2), names[int(class_id)], fill="white", stroke_width=2, stroke_fill="black")
        canvas.thumbnail((1280, 1280))
        canvas.save(preview / f"{relative.parent.name}_{relative.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    split = split_data()
    expected = {part: split[part] for part in ("train", "val", "test")}
    reports = []
    for taxonomy in TAXONOMIES:
        reports.append(validate_taxonomy(taxonomy, expected))
        visualize(taxonomy, args.samples, split["seed"])
    print(*reports, sep="\n")


if __name__ == "__main__":
    main()

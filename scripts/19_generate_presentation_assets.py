from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "annotations.json"
OUTPUT = ROOT / "presentation" / "assets"
IMAGE_ID = 852

PALETTE = [
    "#006633", "#d95f02", "#3561a7", "#8e44ad", "#c0392b",
    "#6a963f", "#d4a017", "#008b8b", "#7f5539", "#e76f51",
]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    candidate = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(candidate), size) if candidate.exists() else ImageFont.load_default()


def polygon_points(
    segmentation: list[list[float]], scale_x: float, scale_y: float
) -> list[list[tuple[float, float]]]:
    polygons: list[list[tuple[float, float]]] = []
    for segment in segmentation:
        if len(segment) >= 6:
            polygons.append([
                (segment[i] * scale_x, segment[i + 1] * scale_y)
                for i in range(0, len(segment), 2)
            ])
    return polygons


def label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, color: str) -> None:
    text_font = font(15, bold=True)
    x, y = xy
    bounds = draw.textbbox((x, y), text, font=text_font, stroke_width=0)
    pad = 3
    draw.rectangle((bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad), fill="white")
    draw.text((x, y), text, font=text_font, fill=color)


def main() -> None:
    data = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    categories = {row["id"]: row["name"] for row in data["categories"]}
    image_info = next(row for row in data["images"] if row["id"] == IMAGE_ID)
    annotations = [row for row in data["annotations"] if row["image_id"] == IMAGE_ID]
    image_path = ROOT / "data" / image_info["file_name"]
    original = Image.open(image_path).convert("RGB")

    annotated_size = (int(image_info["width"]), int(image_info["height"]))
    if abs(original.width / original.height - annotated_size[0] / annotated_size[1]) > 0.01:
        raise ValueError(f"Image aspect ratio {original.size} differs from COCO metadata {annotated_size}")
    scale_x = original.width / annotated_size[0]
    scale_y = original.height / annotated_size[1]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    original.save(OUTPUT / "taco_original_example.png", optimize=True)

    category_ids = sorted({row["category_id"] for row in annotations})
    colors = {category_id: PALETTE[index % len(PALETTE)] for index, category_id in enumerate(category_ids)}

    segmentation = original.copy().convert("RGBA")
    overlay = Image.new("RGBA", segmentation.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay, "RGBA")
    for row in annotations:
        color = colors[row["category_id"]]
        rgb = ImageColor.getrgb(color)
        for polygon in polygon_points(row.get("segmentation", []), scale_x, scale_y):
            overlay_draw.polygon(polygon, fill=(*rgb, 100), outline=(*rgb, 255), width=3)
    segmentation = Image.alpha_composite(segmentation, overlay).convert("RGB")
    segmentation.save(OUTPUT / "taco_segmentation_example.png", optimize=True)

    boxes = original.copy()
    boxes_draw = ImageDraw.Draw(boxes)
    ranked = sorted(annotations, key=lambda row: row["area"], reverse=True)
    label_ids = {row["id"] for row in ranked[:6]}
    for row in annotations:
        color = colors[row["category_id"]]
        x, y, width, height = row["bbox"]
        x, width = x * scale_x, width * scale_x
        y, height = y * scale_y, height * scale_y
        boxes_draw.rectangle((x, y, x + width, y + height), outline=color, width=4)
        if row["id"] in label_ids:
            class_name = categories[row["category_id"]]
            label_y = y + height - 21 if class_name == "Paper cup" else max(2, y + 3)
            label(boxes_draw, (x + 3, label_y), class_name, color)
    boxes.save(OUTPUT / "taco_bbox_example.png", optimize=True)

    manifest = {
        "image_id": IMAGE_ID,
        "file_name": image_info["file_name"],
        "instances": len(annotations),
        "fine_classes": len(category_ids),
        "classes": [categories[category_id] for category_id in category_ids],
        "note": "Segmentation polygons and bounding boxes are read from data/annotations.json and scaled to the downloaded image resolution.",
    }
    (OUTPUT / "taco_annotation_example.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "annotations.json"
TAXONOMIES = ("fine", "material", "binary")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def save_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def config() -> dict[str, Any]:
    with (ROOT / "configs" / "experiment.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def coco() -> dict[str, Any]:
    return load_json(ANNOTATIONS)


def image_path(image: dict[str, Any]) -> Path:
    return ROOT / "data" / Path(image["file_name"])


def image_key(image: dict[str, Any]) -> str:
    return Path(image["file_name"]).as_posix()


def taxonomy_names(data: dict[str, Any], taxonomy: str) -> list[str]:
    if taxonomy == "fine":
        return [c["name"] for c in sorted(data["categories"], key=lambda x: x["id"])]
    if taxonomy == "material":
        return load_json(ROOT / "configs" / "material_mapping.json")["macroclasses"]
    if taxonomy == "binary":
        return ["Litter"]
    raise ValueError(f"Unknown taxonomy: {taxonomy}")


def category_map(data: dict[str, Any], taxonomy: str) -> dict[int, int]:
    categories = sorted(data["categories"], key=lambda x: x["id"])
    if taxonomy == "fine":
        return {c["id"]: i for i, c in enumerate(categories)}
    if taxonomy == "binary":
        return {c["id"]: 0 for c in categories}
    material = load_json(ROOT / "configs" / "material_mapping.json")
    name_to_id = {name: i for i, name in enumerate(material["macroclasses"])}
    return {c["id"]: name_to_id[material["mapping"][c["name"]]["material"]] for c in categories}


def annotations_by_image(data: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        result[annotation["image_id"]].append(annotation)
    return result


def split_data() -> dict[str, Any]:
    path = ROOT / "splits" / "split.json"
    if not path.exists():
        raise FileNotFoundError("Run scripts/02_create_split.py first")
    return load_json(path)


def split_lookup(split: dict[str, Any]) -> dict[str, str]:
    return {name: part for part in ("train", "val", "test") for name in split[part]}


def dataset_dir(taxonomy: str) -> Path:
    return ROOT / "datasets" / f"taco_{taxonomy}"


def label_path(taxonomy: str, part: str, file_name: str) -> Path:
    return dataset_dir(taxonomy) / "labels" / part / Path(file_name).with_suffix(".txt")


def hardlink_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def find_images(root: Path) -> list[Path]:
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in supported)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "python": sys.version,
        "platform": sys.platform,
    }
    try:
        import torch
        metadata.update({
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except ImportError:
        metadata["torch"] = None
    try:
        import ultralytics
        metadata["ultralytics"] = ultralytics.__version__
    except ImportError:
        metadata["ultralytics"] = None
    return metadata

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

from pipeline_utils import ROOT, TAXONOMIES, config, dataset_dir, find_images, runtime_metadata, save_json, seed_everything, taxonomy_names, coco


def smoke_yaml(taxonomy: str, train_count: int, val_count: int) -> Path:
    source = dataset_dir(taxonomy)
    train = find_images(source / "images" / "train")[:train_count]
    val = find_images(source / "images" / "val")[:val_count]
    if not train or not val:
        raise RuntimeError("Built dataset is missing; run scripts 03 and 04 first")
    work = ROOT / "runs" / "smoke" / "manifests"
    work.mkdir(parents=True, exist_ok=True)
    train_txt = work / f"{taxonomy}_train.txt"; val_txt = work / f"{taxonomy}_val.txt"
    train_txt.write_text("\n".join(str(p.resolve()) for p in train) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(str(p.resolve()) for p in val) + "\n", encoding="utf-8")
    descriptor = {"path": str(ROOT.resolve()), "train": str(train_txt.resolve()), "val": str(val_txt.resolve()), "test": str(val_txt.resolve()), "names": {i: n for i, n in enumerate(taxonomy_names(coco(), taxonomy))}}
    target = work / f"{taxonomy}.yaml"
    target.write_text(yaml.safe_dump(descriptor, sort_keys=False), encoding="utf-8")
    return target


def run_root(seed: int, configured_seed: int) -> Path:
    """Keep the original seed-42 layout stable and isolate additional seeds."""
    return ROOT / "runs" if seed == configured_seed else ROOT / "runs" / "seeds" / str(seed)


def train_one(taxonomy: str, smoke: bool, resume: bool, force: bool, seed: int | None = None) -> None:
    cfg = config()
    seed = cfg["seed"] if seed is None else seed
    seed_everything(seed)
    if smoke:
        data = smoke_yaml(taxonomy, 32, 16)
        project, name, epochs, patience = ROOT / "runs" / "smoke", taxonomy, 2, 2
    else:
        data = dataset_dir(taxonomy) / "dataset.yaml"
        project, name, epochs, patience = run_root(seed, cfg["seed"]), taxonomy, cfg["epochs"], cfg["patience"]
    output = project / name
    last = output / "weights" / "last.pt"
    best = output / "weights" / "best.pt"
    metadata_path = output / "experiment_config.json"
    completed = False
    if metadata_path.exists():
        try:
            completed = "training_time_seconds" in json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            completed = False
    if best.exists() and completed and not force:
        print(f"Skip completed run: {best}")
        return
    model = YOLO(str(last if resume and last.exists() else cfg["model"]))
    arguments = {
        "data": str(data), "imgsz": cfg["imgsz"], "epochs": epochs,
        "patience": patience, "batch": cfg["batch"], "device": cfg["device"],
        "workers": cfg["workers"], "seed": seed, "deterministic": cfg["deterministic"],
        "amp": cfg["amp"], "project": str(project), "name": name, "exist_ok": True,
        "optimizer": cfg["optimizer"], "lr0": cfg["lr0"],
        "pretrained": True, "plots": not smoke, "verbose": True, "resume": bool(resume and last.exists()),
    }
    metadata = {**runtime_metadata(), "taxonomy": taxonomy, "smoke": smoke, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in arguments.items()}}
    save_json(output / "experiment_config.json", metadata)
    started = time.perf_counter()
    model.train(**arguments)
    metadata["training_time_seconds"] = time.perf_counter() - started
    save_json(output / "experiment_config.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO11n under a controlled protocol.")
    parser.add_argument("--taxonomy", choices=(*TAXONOMIES, "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun and overwrite an existing completed run.")
    parser.add_argument("--seed", type=int, help="Training seed. The configured seed keeps the legacy output layout.")
    args = parser.parse_args()
    for taxonomy in TAXONOMIES if args.taxonomy == "all" else (args.taxonomy,):
        train_one(taxonomy, args.smoke, args.resume, args.force, args.seed)


if __name__ == "__main__":
    main()

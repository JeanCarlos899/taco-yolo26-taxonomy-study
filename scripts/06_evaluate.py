from __future__ import annotations

import argparse
import math
from pathlib import Path

from ultralytics import YOLO

from evaluation_utils import evaluate_predictions
from pipeline_utils import ROOT, TAXONOMIES, config, dataset_dir, find_images, load_json, save_csv, save_json


def evaluate_one(taxonomy: str, seed: int | None = None) -> None:
    cfg = config()
    seed = cfg["seed"] if seed is None else seed
    run_root = ROOT / "runs" if seed == cfg["seed"] else ROOT / "runs" / "seeds" / str(seed)
    result_root = ROOT / "results" if seed == cfg["seed"] else ROOT / "results" / "seeds" / str(seed)
    weights = run_root / taxonomy / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Missing trained checkpoint: {weights}")
    model = YOLO(str(weights))
    validation = model.val(data=str(dataset_dir(taxonomy) / "dataset.yaml"), split="test", device=0, batch=cfg["batch"], workers=0, plots=True, project=str(run_root / "test"), name=taxonomy, exist_ok=True)
    official = {key: float(value) for key, value in validation.results_dict.items() if isinstance(value, (int, float))}
    test_root = dataset_dir(taxonomy) / "images" / "test"
    sources = find_images(test_root)
    predictions = []
    for result in model.predict(source=[str(p) for p in sources], device=0, conf=0.001, iou=0.7, max_det=300, stream=True, verbose=False):
        source = Path(result.path).resolve(); file_name = source.relative_to(test_root.resolve()).as_posix()
        if result.boxes is None: continue
        for xyxy, confidence, class_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            predictions.append({"file_name": file_name, "class_id": int(class_id), "confidence": float(confidence), "bbox_xyxy": [float(x) for x in xyxy]})
    predictions_path = result_root / "predictions" / f"{taxonomy}.json"
    save_json(predictions_path, predictions)
    aware, per_class = evaluate_predictions(taxonomy, predictions_path, agnostic=False)
    save_json(result_root / f"{taxonomy}_test_metrics.json", {"seed": seed, "ultralytics": official, "controlled_evaluator": aware})
    save_csv(result_root / f"{taxonomy}_class_metrics.csv", per_class, list(per_class[0]))
    print(taxonomy, aware)


def main() -> None:
    parser = argparse.ArgumentParser(description="Final held-out test evaluation.")
    parser.add_argument("--taxonomy", choices=(*TAXONOMIES, "all"), default="all")
    parser.add_argument("--confirm-test", action="store_true", help="Required guard against accidental test-set use.")
    parser.add_argument("--seed", type=int, help="Seed-specific run to evaluate.")
    args = parser.parse_args()
    if not args.confirm_test:
        raise SystemExit("Refusing to evaluate the held-out test set without --confirm-test")
    for taxonomy in TAXONOMIES if args.taxonomy == "all" else (args.taxonomy,):
        evaluate_one(taxonomy, args.seed)


if __name__ == "__main__":
    main()

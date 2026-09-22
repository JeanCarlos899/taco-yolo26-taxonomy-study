from __future__ import annotations

import random

from pipeline_utils import ROOT, coco, config, image_key, save_json


def main() -> None:
    cfg = config()
    images = sorted(image_key(i) for i in coco()["images"])
    random.Random(cfg["seed"]).shuffle(images)
    n_train = int(len(images) * cfg["train_fraction"])
    n_val = int(len(images) * cfg["val_fraction"])
    split = {
        "seed": cfg["seed"],
        "fractions": {"train": cfg["train_fraction"], "val": cfg["val_fraction"], "test": cfg["test_fraction"]},
        "train": sorted(images[:n_train]),
        "val": sorted(images[n_train:n_train + n_val]),
        "test": sorted(images[n_train + n_val:]),
    }
    path = ROOT / "splits" / "split.json"
    if path.exists():
        from pipeline_utils import load_json
        if load_json(path) != split:
            raise RuntimeError(f"Refusing to replace an existing, different split: {path}")
        print(f"Unchanged: {path}")
        return
    save_json(path, split)
    print({part: len(split[part]) for part in ("train", "val", "test")})


if __name__ == "__main__":
    main()

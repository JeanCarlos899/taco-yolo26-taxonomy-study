from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pipeline_utils import ROOT


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script), *arguments]
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def prepare() -> None:
    run("00_download_images.py")
    run("01_analyze_taco.py")
    run("02_create_split.py")
    run("03_build_taxonomies.py")
    run("04_validate_dataset.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible TACO taxonomy-granularity pipeline.")
    parser.add_argument("--stage", choices=("prepare", "smoke", "train", "evaluate", "all"), default="prepare")
    parser.add_argument("--confirm-full-training", action="store_true")
    args = parser.parse_args()
    if args.stage in ("prepare", "all"):
        prepare()
    if args.stage in ("smoke", "all"):
        run("05_train.py", "--smoke")
    if args.stage in ("train", "all"):
        if not args.confirm_full_training:
            raise SystemExit("Full training requires --confirm-full-training after inspecting validation and label previews.")
        run("05_train.py")
    if args.stage in ("evaluate", "all"):
        if not args.confirm_full_training:
            raise SystemExit("Held-out evaluation requires --confirm-full-training.")
        run("06_evaluate.py", "--confirm-test")
        run("07_evaluate_class_agnostic.py")
        run("08_long_tail_analysis.py")
        run("10_generate_tables.py")
        run("09_generate_figures.py")


if __name__ == "__main__":
    main()

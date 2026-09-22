from __future__ import annotations

import argparse
import subprocess
import sys

from pipeline_utils import ROOT, TAXONOMIES, config


def run(command: list[str]) -> None:
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the controlled multi-seed experiment.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--resume", action="store_true", help="Resume interrupted last.pt checkpoints.")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    python = sys.executable
    for seed in args.seeds:
        for taxonomy in TAXONOMIES:
            command = [python, "scripts/05_train.py", "--taxonomy", taxonomy, "--seed", str(seed)]
            if args.resume:
                command.append("--resume")
            run(command)
        if not args.skip_evaluation:
            for taxonomy in TAXONOMIES:
                run([python, "scripts/06_evaluate.py", "--taxonomy", taxonomy, "--seed", str(seed), "--confirm-test"])
    summary_seeds = list(dict.fromkeys([config()["seed"], *args.seeds]))
    run([python, "scripts/15_summarize_multiseed.py", "--seeds", *map(str, summary_seeds)])


if __name__ == "__main__":
    main()

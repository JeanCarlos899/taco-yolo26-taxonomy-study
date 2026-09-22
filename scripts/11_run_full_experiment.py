from __future__ import annotations

import subprocess
import sys
import traceback

from pipeline_utils import ROOT, runtime_metadata, save_json, utc_now


STATUS = ROOT / "results" / "full_run_status.json"


def execute(stage: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--stage", stage, "--confirm-full-training"],
        cwd=ROOT,
        check=True,
    )


def train_with_resume() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "05_train.py"), "--resume"],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    status = {**runtime_metadata(), "state": "training", "started_at_utc": utc_now()}
    save_json(STATUS, status)
    try:
        train_with_resume()
        status.update({"state": "evaluating", "training_finished_at_utc": utc_now()})
        save_json(STATUS, status)
        execute("evaluate")
        status.update({"state": "complete", "finished_at_utc": utc_now()})
        save_json(STATUS, status)
    except BaseException as exc:
        status.update({"state": "failed", "failed_at_utc": utc_now(), "error": repr(exc), "traceback": traceback.format_exc()})
        save_json(STATUS, status)
        raise


if __name__ == "__main__":
    main()

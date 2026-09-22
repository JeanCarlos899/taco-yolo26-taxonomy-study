# Reproducible TACO + YOLO26n pipeline

This project compares the same official TACO images under fine-grained, material-level, and binary taxonomies. The split and training protocol are shared by design. Test evaluation is guarded by an explicit flag.

## Environment (Windows / NVIDIA)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-pipeline.txt
```

On this machine, the preserved Python 3.12 environment used by the unattended paper run is `C:\dev\TACO\.venv312`; replace `.venv` with `..\.venv312` when running commands from the repository directory.

The requirements explicitly select the CUDA 12.8 PyTorch wheels. Do not remove the `+cu128` suffix: the ordinary PyPI wheel is CPU-only on this machine.

The experiment uses `workers=0` on Windows. Spawning multiple data-loader processes duplicates the large CUDA runtime and can exhaust the system page file even when GPU VRAM is available.

All taxonomies use `batch=24`, AdamW, and `lr0=0.001`. These values are fixed explicitly so Ultralytics `optimizer=auto` cannot choose taxonomy-dependent learning rates. Batch 24 uses approximately 4 GB of VRAM on the local RTX 3060 Laptop GPU.

Verify CUDA before training:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Controlled execution

Prepare and validate the data first:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --stage prepare
```

Inspect `results/dataset_statistics.json`, `results/category_distribution.csv`, `configs/material_mapping.json`, `splits/split.json`, and the rendered boxes in `figures/label_previews/`. Then run the two-epoch smoke tests:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --stage smoke
```

Only after those checks, run the three experiments (up to 60 epochs, with early stopping after 15 epochs without validation improvement):

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --stage train --confirm-full-training
```

Evaluate once on the held-out test set and generate analyses, tables, and figures:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --stage evaluate --confirm-full-training
```

To train all three models and automatically run every final analysis afterward:

```powershell
.\.venv\Scripts\python.exe scripts\11_run_full_experiment.py
```

Progress is recorded in `results/full_run_status.json`; training output can be redirected to a log when running unattended.

Use the clean epoch-level monitor below instead of tailing the raw Ultralytics log, which contains ANSI/Unicode progress-bar control characters:

```powershell
.\.venv\Scripts\python.exe scripts\12_watch_training.py
```

`--stage all --confirm-full-training` runs every phase, but the staged commands are preferable because they preserve the intended manual checkpoint before expensive training.

The custom evaluator uses one-to-one greedy matching in descending confidence order and 101-point interpolated AP at IoU 0.50:0.95. It runs both class-aware and class-agnostic evaluations on exactly the same saved model predictions. Ultralytics' native test metrics are also retained in each `results/*_test_metrics.json` file.

## Paper-strengthening analyses

Run the hierarchical re-evaluation, paired image bootstrap (2,000 resamples), semantic-error decomposition, and qualitative examples with:

```powershell
.\.venv\Scripts\python.exe scripts\14_strengthen_analysis.py --bootstrap-iterations 2000
```

This evaluates the exact same Fine-model boxes after remapping their labels to Fine, Material, and Binary. The percentile bootstrap uses the test image as the resampling unit and the same sampled image indices for both sides of every comparison. Outputs are written to `results/fine_hierarchical_evaluation.csv`, `results/bootstrap_comparisons.csv`, `results/fine_error_decomposition.csv`, and `figures/05_error_examples.*`.

Train the controlled additional seeds and evaluate their held-out predictions with:

```powershell
.\.venv\Scripts\python.exe scripts\13_run_multiseed.py --seeds 123 2026 --resume
.\.venv\Scripts\python.exe scripts\15_summarize_multiseed.py --seeds 42 123 2026
```

Seed 42 keeps the original `runs/<taxonomy>` and `results/` layout. Additional seeds are isolated under `runs/seeds/<seed>/` and `results/seeds/<seed>/`. The aggregate tables are `results/multiseed_raw.csv` and `results/multiseed_summary.csv` and report the sample mean and sample standard deviation.

The semantic error decomposition uses a fixed confidence threshold of 0.25 and class-agnostic greedy matching at IoU 0.50. Matched errors are divided into within-material and cross-material confusions. Unmatched predictions with best IoU in `[0.10, 0.50)` are localization failures, lower-overlap predictions are background false positives, and unmatched ground-truth objects are missed detections.

For a concise snapshot of every background run:

```powershell
..\.venv312\Scripts\python.exe scripts\16_monitor_paper_run.py
```

Add `--watch` for a continuously refreshed terminal (10-second interval by default):

```powershell
..\.venv312\Scripts\python.exe scripts\16_monitor_paper_run.py --watch
```

Raw standard output and errors from the unattended jobs are stored in `logs/multiseed.*.log` and `logs/analysis.*.log`.

The official annotation file is never modified. Boxes that extend fractionally beyond an image boundary are clipped only during YOLO conversion and audited in `results/bbox_corrections.json`.

Downloaded JPEGs are physically transposed according to EXIF orientation. Their aspect ratio is checked against the official COCO width/height before they are accepted, preventing rotated pixels from being paired with portrait annotations.

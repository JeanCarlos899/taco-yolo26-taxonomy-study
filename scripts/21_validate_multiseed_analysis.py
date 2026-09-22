#!/usr/bin/env python3
"""Validate the reproducibility and internal consistency of multiseed analyses."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEEDS = (42, 123, 2026)
CATEGORIES = (
    "correct_fine",
    "within_material_confusion",
    "cross_material_confusion",
    "localization_failure",
    "duplicate_detection",
    "background_false_positive",
    "missed_detection",
)


def seed_root(seed: int) -> Path:
    return RESULTS if seed == 42 else RESULTS / "seeds" / str(seed)


def close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=1e-10, atol=1e-12))


def main() -> None:
    raw_errors = pd.read_csv(RESULTS / "multiseed_error_raw.csv")
    error_summary = pd.read_csv(RESULTS / "multiseed_error_summary.csv")
    bootstrap = pd.read_csv(RESULTS / "bootstrap_multiseed.csv")
    bootstrap_summary = pd.read_csv(RESULTS / "bootstrap_multiseed_summary.csv")

    assert set(raw_errors.seed) == set(SEEDS)
    assert set(raw_errors.category) == set(CATEGORIES) | {"semantic_confusion_among_matches"}
    assert len(bootstrap) == len(SEEDS) * 3 * 3
    assert set(bootstrap.seed) == set(SEEDS)

    for seed in SEEDS:
        root = seed_root(seed)
        config = json.loads((root / "bootstrap_config.json").read_text(encoding="utf-8"))
        comparisons = pd.read_csv(root / "bootstrap_comparisons.csv")
        events = json.loads((root / "fine_error_events.json").read_text(encoding="utf-8"))
        counts = Counter(event["category"] for event in events)

        assert config["iterations"] == 2000
        # The original seed-42 artifact predates the explicit test_images field.
        # Its event inventory below independently verifies the same 637 instances.
        assert config.get("test_images", 225) == 225
        assert len(comparisons) == 9
        assert set(comparisons.iterations) == {2000}
        assert set(counts) == set(CATEGORIES)
        assert sum(counts[name] for name in CATEGORIES[:3]) + counts["missed_detection"] == 637

        for row in comparisons.itertuples():
            assert row.ci95_low <= row.bootstrap_mean_difference <= row.ci95_high
            assert bool(row.includes_zero) == (row.ci95_low <= 0 <= row.ci95_high)

        reported = raw_errors[raw_errors.seed == seed].set_index("category")
        spatial = sum(counts[name] for name in CATEGORIES[:3])
        predictions = sum(counts[name] for name in CATEGORIES[:-1])
        for category in CATEGORIES:
            assert int(reported.loc[category, "count"]) == counts[category]
        assert int(reported.loc["semantic_confusion_among_matches", "count"]) == (
            counts["within_material_confusion"] + counts["cross_material_confusion"]
        )
        assert close(reported.loc["correct_fine", "rate"], counts["correct_fine"] / spatial)
        assert close(reported.loc["duplicate_detection", "rate"], counts["duplicate_detection"] / predictions)
        assert close(reported.loc["missed_detection", "rate"], counts["missed_detection"] / 637)

    for row in error_summary.itertuples():
        group = raw_errors[raw_errors.category == row.category]
        assert len(group) == len(SEEDS)
        assert close(row.mean_count, group["count"].mean())
        assert close(row.std_count, group["count"].std(ddof=1))
        assert close(row.mean_rate, group.rate.mean())
        assert close(row.std_rate, group.rate.std(ddof=1))

    for row in bootstrap_summary.itertuples():
        group = bootstrap[(bootstrap.comparison == row.comparison) & (bootstrap.metric == row.metric)]
        excludes = ~group.includes_zero.astype(bool)
        directions = set(group.direction)
        assert len(group) == len(SEEDS)
        assert close(row.mean_difference_across_seeds, group.bootstrap_mean_difference.mean())
        assert close(row.std_difference_across_seeds, group.bootstrap_mean_difference.std(ddof=1))
        assert row.seeds_excluding_zero == int(excludes.sum())
        assert bool(row.consistent_direction) == (len(directions) == 1)
        assert bool(row.all_exclude_zero_same_direction) == (excludes.all() and len(directions) == 1)

    candidates = json.loads(
        (RESULTS / "multiseed_qualitative_candidates.json").read_text(encoding="utf-8")
    )
    assert set(candidates) == {"stable_correct", "stable_semantic_error", "seed_disagreement"}
    for items in candidates.values():
        assert len(items) <= 30
        for item in items:
            assert set(map(int, item["outcomes"])) == set(SEEDS)

    print("Multiseed analysis validation passed: 3 seeds, 2,000 bootstrap replicates each.")


if __name__ == "__main__":
    main()

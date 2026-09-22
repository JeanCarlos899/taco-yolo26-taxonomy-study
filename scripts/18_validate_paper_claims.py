from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from pipeline_utils import ROOT


PAPER = ROOT / "paper"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def text_corpus() -> str:
    files = [PAPER / "main.tex", *sorted((PAPER / "sections").glob("*.tex")),
             *sorted((PAPER / "tables").glob("*.tex"))]
    return "\n".join(path.read_text(encoding="utf-8") for path in files)


def main() -> None:
    corpus = text_corpus()
    facts = json.loads((PAPER / "article_facts.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    native = pd.read_csv(ROOT / "results" / "multiseed_summary.csv").set_index("taxonomy")
    hierarchy = pd.read_csv(ROOT / "results" / "hierarchical_multiseed_summary.csv").set_index("evaluation_taxonomy")
    bootstrap = pd.read_csv(ROOT / "results" / "bootstrap_multiseed_summary.csv")
    errors = pd.read_csv(ROOT / "results" / "multiseed_error_raw.csv")

    require(facts["dataset"] == {"images": 1500, "objects": 4784, "fine_classes": 60,
                                 "images_without_objects": 0, "boxes_clipped_during_conversion": 7},
            "Dataset facts differ from the article assumptions.")
    require(facts["split_images"] == {"train": 1050, "val": 225, "test": 225}, "Image split changed.")
    require(facts["split_objects"] == {"train": 3359, "val": 788, "test": 637}, "Object split changed.")
    require([config[key] for key in ("imgsz", "epochs", "patience", "batch")] == [640, 60, 15, 24],
            "Training configuration changed.")

    expected_native = {"fine": .146, "material": .183, "binary": .567}
    for taxonomy, expected in expected_native.items():
        require(round(float(native.loc[taxonomy, "map50_mean"]), 3) == expected,
                f"Native AP50 changed for {taxonomy}.")
    expected_hierarchy = {"fine": .103, "material": .171, "binary": .466}
    for taxonomy, expected in expected_hierarchy.items():
        require(round(float(hierarchy.loc[taxonomy, "ap50_mean"]), 3) == expected,
                f"Hierarchical AP50 changed for {taxonomy}.")

    expected_support = {
        ("fine_direct_minus_material_direct", "ap50"): 1,
        ("fine_direct_minus_material_direct", "map50_95"): 1,
        ("fine_direct_minus_material_direct", "f1"): 3,
        ("fine_as_material_minus_fine_direct", "ap50"): 3,
        ("fine_as_material_minus_fine_direct", "map50_95"): 2,
        ("fine_as_material_minus_fine_direct", "f1"): 3,
        ("fine_as_binary_minus_fine_direct", "ap50"): 3,
        ("fine_as_binary_minus_fine_direct", "map50_95"): 3,
        ("fine_as_binary_minus_fine_direct", "f1"): 3,
    }
    observed_support = {(row.comparison, row.metric): int(row.seeds_excluding_zero)
                        for row in bootstrap.itertuples()}
    require(observed_support == expected_support, "Multiseed bootstrap support changed.")
    expected_errors = {
        42: [97, 51, 73, 11, 32, 42, 416],
        123: [106, 57, 79, 16, 42, 48, 395],
        2026: [97, 52, 85, 22, 48, 52, 403],
    }
    categories = ["correct_fine", "within_material_confusion", "cross_material_confusion",
                  "localization_failure", "duplicate_detection", "background_false_positive",
                  "missed_detection"]
    for seed, expected in expected_errors.items():
        observed = errors[errors.seed == seed].set_index("category")["count"]
        require([int(observed[name]) for name in categories] == expected,
                f"Error decomposition changed for seed {seed}.")

    required_fragments = ["1.500 imagens", "4.784 objetos", "1.050/225/225", "3.359/788/637",
                          "0,146", "0,183", "0,567", "0{,}103", "0{,}171", "0{,}466",
                          "2.000 reamostragens", "56{,}95", "404{,}7", "63{,}5",
                          "Fine$\\rightarrow$Material", "Fine$\\rightarrow$Binary", "1/3", "2/3", "3/3"]
    missing = [item for item in required_fragments if item not in corpus]
    require(not missing, f"Expected factual fragments missing from TeX: {missing}")

    required_assets = ["methodology_pipeline.pdf", "class_rank_distribution.pdf",
                       "taxonomy_and_hierarchy.pdf", "qualitative_analysis.pdf",
                       "model_level_examples.pdf"]
    require(all((PAPER / "figures" / name).exists() for name in required_assets), "A paper figure is missing.")

    report = {
        "status": "passed",
        "checks": {
            "dataset_and_split": True,
            "training_configuration": True,
            "native_multiseed_metrics": True,
            "hierarchical_multiseed_metrics": True,
            "bootstrap_multiseed_interpretation": True,
            "error_decomposition_multiseed": True,
            "tex_factual_fragments": True,
            "required_figures": True,
        },
    }
    (PAPER / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Paper claim validation passed.")


if __name__ == "__main__":
    main()

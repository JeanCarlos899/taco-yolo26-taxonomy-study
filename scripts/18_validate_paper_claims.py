from __future__ import annotations

import json
import re
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
    controlled = pd.read_csv(ROOT / "results" / "controlled_multiseed_summary.csv").set_index("taxonomy")
    controlled_raw = pd.read_csv(ROOT / "results" / "controlled_multiseed_raw.csv")
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
    require(len(controlled_raw) == 9 and len(controlled_raw[['seed', 'taxonomy']].drop_duplicates()) == 9,
            "Unified evaluation must contain all nine distinct runs.")
    for taxonomy in ('fine', 'material', 'binary'):
        current = controlled_raw[controlled_raw.taxonomy == taxonomy]
        require(set(current.seed) == {42, 123, 2026}, "A seed is missing from unified evaluation.")
        for metric in ('precision', 'recall', 'f1', 'map50', 'map50_95'):
            require(abs(float(current[metric].mean()) - float(controlled.loc[taxonomy, metric + '_mean'])) < 1e-12,
                    "Unified summary does not match raw results.")
            require(abs(float(current[metric].std(ddof=1)) - float(controlled.loc[taxonomy, metric + '_std'])) < 1e-12,
                    "Unified standard deviation does not match raw results.")
    for metric, hierarchical_metric in [('precision', 'precision'), ('recall', 'recall'), ('f1', 'f1'),
                                       ('map50', 'ap50'), ('map50_95', 'map50_95')]:
        require(abs(float(controlled.loc['fine', metric + '_mean']) - float(hierarchy.loc['fine', hierarchical_metric + '_mean'])) < 1e-12,
                "Fine baseline differs between the main and hierarchical tables.")
    main_table = (PAPER / 'tables/main_results.tex').read_text(encoding='utf-8')
    for taxonomy in ('fine', 'material', 'binary'):
        row = next(line for line in main_table.splitlines() if line.startswith(taxonomy.capitalize() + ' &'))
        expected = [f"{controlled.loc[taxonomy, metric + '_mean']:.3f}$\\pm${controlled.loc[taxonomy, metric + '_std']:.3f}"
                    for metric in ('precision', 'recall', 'f1', 'map50', 'map50_95')]
        require(all(value in row for value in expected), "Main table contains non-unified values.")

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
                          "0,103", "0,156", "0,566", "0{,}103", "0{,}171", "0{,}466",
                          "2.000 reamostragens", "56{,}95", "404{,}7", "63{,}5",
                          "Fine como Material", "Fine como Binary", "1/3", "2/3", "3/3",
                          "poucas classes de ``cabeça''", "muitas classes da ``cauda''",
                          "estratos operacionais deste estudo"]
    missing = [item for item in required_fragments if item not in corpus]
    require(not missing, f"Expected factual fragments missing from TeX: {missing}")
    required_identity = ["Jean Carlos Rodrigues Sousa", "Karielly de Carvalho",
                         "Marcos A. G. B. Brito",
                         "Universidade Federal do Piauí (UFPI)",
                         "Campus Senador Helvídio Nunes de Barros (CSHNB)",
                         "Picos -- PI -- Brasil", "jean.rodrigues", "karielly.carvalho",
                         "marcos.brito", "@ufpi.edu.br"]
    require(all(item in corpus for item in required_identity), "Author or affiliation data is incomplete.")
    require("email@exemplo.com" not in corpus and "Nome do(a) Autor(a)" not in corpus,
            "Placeholder identity remains in the paper.")

    first_person_pattern = re.compile(
        r"\b(?:nós|nosso|nossa|nossos|nossas|usamos|mantemos|derivamos|treinamos|"
        r"aplicamos|reportamos|fizemos|orientamos|contamos|selecionamos|empregamos|"
        r"fixamos|medimos|alteramos|investigamos|confirmamos|reutilizamos|avaliamos|"
        r"comparamos|definimos|mostramos|propomos|adotamos)\b",
        flags=re.IGNORECASE,
    )
    require(first_person_pattern.search(corpus) is None,
            "First-person authorial language remains in the paper.")

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
            "unified_multiseed_metrics": True,
            "identical_fine_baseline": True,
            "hierarchical_multiseed_metrics": True,
            "bootstrap_multiseed_interpretation": True,
            "error_decomposition_multiseed": True,
            "tex_factual_fragments": True,
            "author_and_affiliation": True,
            "impersonal_academic_style": True,
            "required_figures": True,
        },
    }
    (PAPER / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Paper claim validation passed.")


if __name__ == "__main__":
    main()

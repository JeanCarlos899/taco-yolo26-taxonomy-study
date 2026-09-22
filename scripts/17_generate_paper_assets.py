from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from evaluation_utils import evaluate_predictions
from pipeline_utils import ROOT, category_map, coco, load_json, save_csv, save_json, taxonomy_names


SEEDS = (42, 123, 2026)
TAXONOMIES = ("fine", "material", "binary")
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"

BLUE = "#242424"
TEAL = "#f2f2f2"
ORANGE = "#666666"
GOLD = "#c9c9c9"
DARK = "#111111"


def result_root(seed: int) -> Path:
    return ROOT / "results" if seed == 42 else ROOT / "results" / "seeds" / str(seed)


def fine_to_material() -> dict[int, int]:
    data = coco()
    fine = category_map(data, "fine")
    material = category_map(data, "material")
    return {fine[key]: material[key] for key in fine}


def hierarchical_multiseed() -> pd.DataFrame:
    mapping = fine_to_material()
    rows = []
    for seed in SEEDS:
        root = result_root(seed)
        source = load_json(root / "predictions" / "fine.json")
        for target in TAXONOMIES:
            if target == "fine":
                converted = source
            elif target == "material":
                converted = [{**item, "class_id": mapping[item["class_id"]]} for item in source]
            else:
                converted = [{**item, "class_id": 0} for item in source]
            path = root / "predictions" / f"fine_as_{target}.json"
            save_json(path, converted)
            metrics, _ = evaluate_predictions(target, path, agnostic=False)
            rows.append({"seed": seed, "evaluation_taxonomy": target,
                         **{key: metrics[key] for key in ("precision", "recall", "f1", "ap50", "map50_95")}})
    save_csv(ROOT / "results" / "hierarchical_multiseed_raw.csv", rows, list(rows[0]))
    frame = pd.DataFrame(rows)
    summary = []
    for taxonomy in TAXONOMIES:
        current = frame[frame.evaluation_taxonomy == taxonomy]
        row = {"evaluation_taxonomy": taxonomy, "num_seeds": len(current)}
        for metric in ("precision", "recall", "f1", "ap50", "map50_95"):
            row[f"{metric}_mean"] = current[metric].mean()
            row[f"{metric}_std"] = current[metric].std(ddof=1)
        summary.append(row)
    save_csv(ROOT / "results" / "hierarchical_multiseed_summary.csv", summary, list(summary[0]))
    return pd.DataFrame(summary)


def long_tail_multiseed() -> pd.DataFrame:
    distribution = pd.read_csv(ROOT / "results" / "class_distribution.csv")
    rows = []
    for seed in SEEDS:
        metrics = pd.read_csv(result_root(seed) / "fine_class_metrics.csv")
        merged = distribution[["class_id", "frequency_group", "train_instances"]].merge(metrics, on="class_id")
        for group in ("Rare", "Medium", "Frequent"):
            current = merged[merged.frequency_group == group]
            rows.append({"seed": seed, "frequency_group": group,
                         "num_classes": int(len(current)),
                         "mean_train_instances": current.train_instances.mean(),
                         "mean_ap50": current.ap50.mean(),
                         "mean_map50_95": current.map50_95.mean(),
                         "mean_recall": current.recall.mean()})
    save_csv(ROOT / "results" / "long_tail_multiseed_raw.csv", rows, list(rows[0]))
    frame = pd.DataFrame(rows)
    summary = []
    for group in ("Rare", "Medium", "Frequent"):
        current = frame[frame.frequency_group == group]
        row = {"frequency_group": group, "num_classes": int(current.num_classes.iloc[0]),
               "mean_train_instances": current.mean_train_instances.mean()}
        for metric in ("mean_ap50", "mean_map50_95", "mean_recall"):
            row[f"{metric}_mean"] = current[metric].mean()
            row[f"{metric}_std"] = current[metric].std(ddof=1)
        summary.append(row)
    save_csv(ROOT / "results" / "long_tail_multiseed_summary.csv", summary, list(summary[0]))
    return pd.DataFrame(summary)


def latex_tables(hierarchy: pd.DataFrame, long_tail: pd.DataFrame) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(ROOT / "results" / "multiseed_summary.csv").set_index("taxonomy")
    display = {"fine": "Fine", "material": "Material", "binary": "Binary"}
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Taxonomia & $P$ & $R$ & $F_1$ & AP$_{50}$ & AP$_{50:95}$ \\", r"\midrule"]
    for taxonomy in TAXONOMIES:
        row = main.loc[taxonomy]
        values = [f"{row[f'{metric}_mean']:.3f}$\\pm${row[f'{metric}_std']:.3f}"
                  for metric in ("precision", "recall", "f1", "map50", "map50_95")]
        lines.append(display[taxonomy] + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "main_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Nível de avaliação & $F_1$ & AP$_{50}$ & AP$_{50:95}$ \\", r"\midrule"]
    for taxonomy in TAXONOMIES:
        row = hierarchy[hierarchy.evaluation_taxonomy == taxonomy].iloc[0]
        values = [f"{row[f'{metric}_mean']:.3f}$\\pm${row[f'{metric}_std']:.3f}"
                  for metric in ("f1", "ap50", "map50_95")]
        lines.append(display[taxonomy] + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "hierarchical_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Grupo & \#classes & $n_{treino}$ & AP$_{50}$ & Recall \\", r"\midrule"]
    translations = {"Rare": "Raras", "Medium": "Médias", "Frequent": "Frequentes"}
    for group in ("Rare", "Medium", "Frequent"):
        row = long_tail[long_tail.frequency_group == group].iloc[0]
        lines.append(f"{translations[group]} & {int(row.num_classes)} & {row.mean_train_instances:.1f} & "
                     f"{row.mean_ap50_mean:.3f}$\\pm${row.mean_ap50_std:.3f} & "
                     f"{row.mean_recall_mean:.3f}$\\pm${row.mean_recall_std:.3f}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "long_tail_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def methodology_figure() -> None:
    """Create a publication-native vector diagram of the full experimental design."""
    fig, ax = plt.subplots(figsize=(7.25, 3.15))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, title, subtitle, color, text_color="white"):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                               facecolor=color, edgecolor="none")
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * .63, title, ha="center", va="center",
                fontsize=9, fontweight="bold", color=text_color)
        ax.text(x + w / 2, y + h * .31, subtitle, ha="center", va="center",
                fontsize=7.2, color=text_color, linespacing=1.2)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                     linewidth=1.15, color="#555555"))

    box(.15, 2.25, 1.65, 1.55, "TACO oficial", "1.500 imagens\n4.784 objetos", BLUE)
    box(2.25, 2.25, 1.75, 1.55, "Auditoria", "caixas, EXIF e\nintegridade", "#555555")
    box(4.45, 2.25, 1.65, 1.55, "Partição fixa", "70/15/15\npor imagem", "#7a7a7a")
    arrow(1.80, 3.02, 2.25, 3.02); arrow(4.00, 3.02, 4.45, 3.02)

    branch_x = 6.7
    for y, title, sub, color in ((4.35, "Fine", "60 classes", ORANGE),
                                  (2.45, "Material", "6 classes", GOLD),
                                  (.55, "Binary", "1 classe", "#8c8c8c")):
        box(branch_x, y, 1.55, 1.05, title, sub, color, DARK if color == GOLD else "white")
        arrow(6.10, 3.02, branch_x, y + .52)

    box(8.75, 2.25, 1.45, 1.55, "YOLO26n", "mesmo protocolo\n3 sementes", DARK)
    for y in (4.87, 2.97, 1.07):
        arrow(8.25, y, 8.75, 3.02)
    box(10.65, 2.25, 1.20, 1.55, "Teste", "225 imagens\nfixas", BLUE)
    arrow(10.20, 3.02, 10.65, 3.02)

    ax.text(9.95, .38, "Avaliação nativa  •  hierárquica  •  bootstrap pareado  •  cauda longa  •  erros",
            ha="center", va="center", fontsize=7.5, color=DARK,
            bbox=dict(boxstyle="round,pad=.35", facecolor="white", edgecolor="#777777"))
    arrow(11.25, 2.25, 10.35, .72)
    fig.tight_layout(pad=.15)
    fig.savefig(FIGURES / "methodology_pipeline.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "methodology_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _crop_for_boxes(image: Image.Image, boxes: list[list[float]], margin: float = .42):
    boxes = [b for b in boxes if b]
    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    span = max(x2 - x1, y2 - y1, min(image.size) * .22)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = span * (0.5 + margin)
    left, top = max(0, cx - half), max(0, cy - half)
    right, bottom = min(image.width, cx + half), min(image.height, cy + half)
    return image.crop((left, top, right, bottom)), (left, top)


def _draw_event(ax, event, title: str, show_gt=True, show_pred=True, raw=False, gt_override=None):
    image = Image.open(ROOT / "datasets" / "taco_fine" / "images" / "test" / event["file_name"]).convert("RGB")
    gt = gt_override if gt_override is not None else event.get("gt_box")
    pred = event.get("pred_box")
    crop, (left, top) = _crop_for_boxes(image, [gt, pred])
    ax.imshow(crop)
    if not raw:
        for box_data, color, label, linestyle in ((gt, "white", "GT", "-"),
                                                   (pred, "black", "Pred.", "--")):
            enabled = (label == "GT" and show_gt) or (label == "Pred." and show_pred)
            if box_data and enabled:
                x1, y1, x2, y2 = box_data
                ax.add_patch(Rectangle((x1-left, y1-top), x2-x1, y2-y1,
                                       fill=False, linewidth=2.6, edgecolor=color, linestyle=linestyle))
                ax.text(x1-left + 2, max(2, y1-top + 2), label, ha="left", va="top", fontsize=7.2,
                        fontweight="bold", color="black" if label == "GT" else "white",
                        bbox=dict(boxstyle="round,pad=.18", facecolor=color, edgecolor="none", alpha=.95))
    ax.set_title(title, fontsize=8.2, fontweight="bold", color=DARK, pad=4)
    ax.axis("off")


def qualitative_figure() -> None:
    events = load_json(ROOT / "results" / "fine_error_events.json")

    def event(category, file_name):
        return next(row for row in events if row["category"] == category and row["file_name"] == file_name)

    correct = event("correct_fine", "batch_3/IMG_4915.JPG")
    within = event("within_material_confusion", "batch_1/000050.jpg")
    cross = event("cross_material_confusion", "batch_3/IMG_4926.JPG")
    loc = max((row for row in events if row["category"] == "localization_failure" and
               row["file_name"] == "batch_14/000047.jpg"), key=lambda row: row.get("confidence") or 0)
    missed = [row for row in events if row["category"] == "missed_detection" and
              row["file_name"] == loc["file_name"] and row.get("gt_box")]
    loc_gt = max((row["gt_box"] for row in missed), key=lambda box: _iou(box, loc["pred_box"]))

    fig, axes = plt.subplots(2, 3, figsize=(7.25, 5.15))
    _draw_event(axes[0, 0], correct, "(a) Entrada", raw=True)
    _draw_event(axes[0, 1], correct, "(b) Verdade-terreno", show_gt=True, show_pred=False)
    _draw_event(axes[0, 2], correct, "(c) Saída correta", show_gt=False, show_pred=True)
    _draw_event(axes[1, 0], within, "(d) Intra-material: lata → aerossol")
    _draw_event(axes[1, 1], cross, "(e) Inter-material: metal → papel")
    _draw_event(axes[1, 2], loc, "(f) Localização insuficiente", gt_override=loc_gt)
    fig.text(.5, .015, "Branco contínuo: verdade-terreno   •   Preto tracejado: predição (semente 42)",
             ha="center", fontsize=7.5, color=DARK)
    fig.tight_layout(rect=(0, .035, 1, 1), pad=.45)
    fig.savefig(FIGURES / "qualitative_analysis.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "qualitative_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figures(hierarchy: pd.DataFrame, long_tail: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    methodology_figure()
    distribution = pd.read_csv(ROOT / "results" / "class_distribution.csv").sort_values("train_instances", ascending=False)
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ranks = np.arange(1, len(distribution) + 1)
    ax.plot(ranks, distribution.train_instances, marker="o", markersize=3, linewidth=1.4, color="black")
    ax.axhline(19, color="#444444", linestyle="--", linewidth=1, label="Rara: $n\\leq19$")
    ax.axhline(100, color="#777777", linestyle=":", linewidth=1.4, label="Frequente: $n>100$")
    ax.set_yscale("symlog", linthresh=1); ax.set_xlabel("Posição da classe por frequência")
    ax.set_ylabel("Instâncias de treino (escala log)"); ax.grid(alpha=.25); ax.legend(frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(FIGURES / "class_rank_distribution.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "class_rank_distribution.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    main = pd.read_csv(ROOT / "results" / "multiseed_summary.csv").set_index("taxonomy")
    labels = ["Fine", "Material", "Binary"]
    x = np.arange(3); width = .34
    for offset, metric, title, color, hatch in ((-.17, "map50", r"AP$_{50}$", "#333333", ""),
                                                (.17, "map50_95", r"AP$_{50:95}$", "#bdbdbd", "///")):
        axes[0].bar(x + offset, [main.loc[t, f"{metric}_mean"] for t in TAXONOMIES], width,
                    yerr=[main.loc[t, f"{metric}_std"] for t in TAXONOMIES], capsize=3,
                    label=title, color=color, edgecolor="black", linewidth=.6, hatch=hatch)
    axes[0].set_xticks(x, labels); axes[0].set_ylim(0, .65); axes[0].set_ylabel("Métrica")
    axes[0].set_title("Modelos treinados por taxonomia"); axes[0].legend(frameon=False); axes[0].grid(axis="y", alpha=.25)
    hierarchy = hierarchy.set_index("evaluation_taxonomy")
    for offset, metric, title, color, hatch in ((-.17, "ap50", r"AP$_{50}$", "#333333", ""),
                                                (.17, "map50_95", r"AP$_{50:95}$", "#bdbdbd", "///")):
        axes[1].bar(x + offset, [hierarchy.loc[t, f"{metric}_mean"] for t in TAXONOMIES], width,
                    yerr=[hierarchy.loc[t, f"{metric}_std"] for t in TAXONOMIES], capsize=3,
                    label=title, color=color, edgecolor="black", linewidth=.6, hatch=hatch)
    axes[1].set_xticks(x, labels); axes[1].set_ylim(0, .65); axes[1].set_title("Mesmo modelo Fine remapeado")
    axes[1].grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(FIGURES / "taxonomy_and_hierarchy.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "taxonomy_and_hierarchy.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 2.8)); x = np.arange(3); width = .34
    for offset, metric, title, color, hatch in ((-.17, "mean_ap50", r"AP$_{50}$", "#333333", ""),
                                                (.17, "mean_recall", "Recall", "#bdbdbd", "///")):
        ax.bar(x + offset, long_tail[f"{metric}_mean"], width, yerr=long_tail[f"{metric}_std"],
               capsize=3, label=title, color=color, edgecolor="black", linewidth=.6, hatch=hatch)
    ax.set_xticks(x, ["Raras", "Médias", "Frequentes"]); ax.set_ylim(0, .4); ax.set_ylabel("Média macro")
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(FIGURES / "long_tail.pdf", bbox_inches="tight"); fig.savefig(FIGURES / "long_tail.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    qualitative_figure()


def manifest() -> None:
    inputs = [ROOT / "configs/experiment.yaml", ROOT / "configs/material_mapping.json",
              ROOT / "splits/split.json", ROOT / "results/multiseed_raw.csv",
              ROOT / "results/bootstrap_comparisons.csv", ROOT / "results/fine_error_decomposition.csv",
              ROOT / "results/class_distribution.csv"]
    records = []
    for path in inputs:
        records.append({"path": path.relative_to(ROOT).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    save_json(PAPER / "results_manifest.json", {"generator": "scripts/17_generate_paper_assets.py", "inputs": records})


def article_facts() -> None:
    data = coco()
    split = load_json(ROOT / "splits" / "split.json")
    membership = {name: part for part in ("train", "val", "test") for name in split[part]}
    images = {row["id"]: Path(row["file_name"]).as_posix() for row in data["images"]}
    objects = {part: 0 for part in ("train", "val", "test")}
    for annotation in data["annotations"]:
        objects[membership[images[annotation["image_id"]]]] += 1
    material = load_json(ROOT / "configs" / "material_mapping.json")
    material_counts = {name: 0 for name in material["macroclasses"]}
    for category in data["categories"]:
        material_counts[material["mapping"][category["name"]]["material"]] += 1
    audit = load_json(ROOT / "results" / "dataset_statistics.json")
    save_json(PAPER / "article_facts.json", {
        "dataset": {"images": len(data["images"]), "objects": len(data["annotations"]),
                    "fine_classes": len(data["categories"]), "images_without_objects": audit["images_without_objects"],
                    "boxes_clipped_during_conversion": audit["invalid_bounding_boxes"]},
        "split_images": {part: len(split[part]) for part in ("train", "val", "test")},
        "split_objects": objects,
        "material_macroclasses": material_counts,
        "binary_class": "Litter",
        "seeds": list(SEEDS),
    })


def main() -> None:
    hierarchy = hierarchical_multiseed()
    long_tail = long_tail_multiseed()
    latex_tables(hierarchy, long_tail)
    figures(hierarchy, long_tail)
    manifest()
    article_facts()
    print("Paper tables, figures, and provenance manifest generated.")


if __name__ == "__main__":
    main()

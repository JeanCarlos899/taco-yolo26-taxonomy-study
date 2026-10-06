"""Reevaluate saved predictions with the same evaluator used by the hierarchy.

No inference or training. Native metrics are retained as separate historical
artifacts; controlled metrics are the primary reporting protocol.
"""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

from evaluation_utils import evaluate_predictions

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2026)
LEVELS = ('fine', 'material', 'binary')
METRICS = ('precision', 'recall', 'f1', 'map50', 'map50_95')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    with (ROOT / 'results/hierarchical_multiseed_raw.csv').open(encoding='utf-8', newline='') as stream:
        hierarchy = {(int(row['seed']), row['evaluation_taxonomy']): row for row in csv.DictReader(stream)}
    rows, inputs = [], []
    for seed in SEEDS:
        root = ROOT / ('results' if seed == 42 else f'results/seeds/{seed}')
        for level in LEVELS:
            path = root / f'predictions/{level}.json'
            metrics, _ = evaluate_predictions(level, path)
            recorded = json.loads((root / f'{level}_test_metrics.json').read_text(encoding='utf-8'))['controlled_evaluator']
            for metric in METRICS:
                assert abs(metrics[metric] - recorded[metric]) < 1e-12, (seed, level, metric)
            assert metrics['test_images'] == 225 and metrics['test_instances'] == 637
            if level == 'fine':
                for metric in ('precision', 'recall', 'f1', 'ap50', 'map50_95'):
                    assert abs(metrics[metric] - float(hierarchy[seed, 'fine'][metric])) < 1e-12
            rows.append({'seed': seed, 'taxonomy': level, **{metric: metrics[metric] for metric in METRICS}})
            inputs.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            print(f'{seed} {level}: AP50={metrics["map50"]:.6f} F1={metrics["f1"]:.6f}', flush=True)
    write_csv(ROOT / 'results/controlled_multiseed_raw.csv', rows)
    summary = []
    for level in LEVELS:
        current = [row for row in rows if row['taxonomy'] == level]
        item = {'taxonomy': level, 'num_seeds': len(current)}
        for metric in METRICS:
            values = [row[metric] for row in current]
            item[f'{metric}_mean'] = mean(values)
            item[f'{metric}_std'] = stdev(values)
        summary.append(item)
    write_csv(ROOT / 'results/controlled_multiseed_summary.csv', summary)
    def display(avg, sd):
        return f'{avg:.3f}'.replace('.', '{,}') + r'\pm' + f'{sd:.3f}'.replace('.', '{,}')
    tex = [r'\begin{tabular}{lcc}', r'\toprule', r'\textbf{Modelo} & \textbf{AP$_{50}$} & \textbf{$F_1$} \\', r'\midrule']
    for row, label in zip(summary, ('Fine (60 classes)', 'Material (6 classes)', 'Binary (1 classe)')):
        tex.append(f'{label} & ${display(row["map50_mean"], row["map50_std"])}$ & ${display(row["f1_mean"], row["f1_std"])}$ ' + r'\\')
    tex.extend([r'\bottomrule', r'\end{tabular}'])
    (ROOT / 'presentation/assets/controlled_main_results.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    audit = {'protocol': 'controlled evaluator, shared with hierarchy and bootstrap', 'new_training': False,
             'new_inference': False, 'inputs': inputs, 'checks': ['All nine evaluations reproduce stored metrics to 1e-12',
             '225 images and 637 objects in every evaluation', 'Fine equals hierarchical baseline for every seed to 1e-12'],
             'summary': summary}
    (ROOT / 'results/unified_evaluation_audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

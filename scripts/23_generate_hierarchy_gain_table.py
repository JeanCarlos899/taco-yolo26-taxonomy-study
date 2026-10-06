"""Report paired changes within the controlled evaluator, never across protocols."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'results/hierarchical_multiseed_raw.csv'
    with source.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    scores = {(int(row['seed']), row['evaluation_taxonomy']): float(row['ap50']) for row in rows}
    expected = {(seed, level) for seed in (42, 123, 2026) for level in ('fine', 'material', 'binary')}
    assert set(scores) == expected and len(rows) == 9, 'Incomplete or duplicated seed results.'
    summary_path = ROOT / 'results/hierarchical_multiseed_summary.csv'
    with summary_path.open(encoding='utf-8', newline='') as stream:
        summary = {row['evaluation_taxonomy']: row for row in csv.DictReader(stream)}
    for level in ('fine', 'material', 'binary'):
        assert abs(mean(scores[seed, level] for seed in (42, 123, 2026)) - float(summary[level]['ap50_mean'])) < 1e-12
    output = []
    tex = [r'\begin{tabular}{lc}', r'\toprule', r'\textbf{Nova exigência} & \textbf{Ganho de AP$_{50}$} \\', r'\midrule']
    labels = {'material': 'Aceitar o material correto', 'binary': 'Aceitar qualquer resíduo localizado'}
    for level in ('material', 'binary'):
        differences = [scores[seed, level] - scores[seed, 'fine'] for seed in (42, 123, 2026)]
        assert all(value > 0 for value in differences)
        average, sd = mean(differences), stdev(differences)
        output.append({'target': level, 'num_seeds': 3, 'ap50_gain_mean': average,
                       'ap50_gain_std': sd, **{f'gain_seed_{seed}': value for seed, value in zip((42, 123, 2026), differences)}})
        value = f'+{average:.3f}'.replace('.', '{,}')
        deviation = f'{sd:.3f}'.replace('.', '{,}')
        tex.append(f'{labels[level]} & ${value}\\pm{deviation}$ ' + r'\\')
    with (ROOT / 'results/presentation_hierarchy_gains.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    tex.extend([r'\bottomrule', r'\end{tabular}'])
    (ROOT / 'presentation/assets/hierarchy_gains.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    audit = {'source': source.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
             'definition': 'Per-seed AP50(target) minus AP50(Fine), both from controlled evaluator.',
             'dispersion': 'Sample standard deviation of the three paired observed differences, not bootstrap CI.',
             'checks': ['Nine unique seed/level results', 'Means match hierarchical summary to 1e-12', 'All six gains positive'],
             'results': output}
    (ROOT / 'results/presentation_hierarchy_gain_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()

import argparse
import json
import os
from collections import defaultdict

import numpy as np


DATASET_CHANNELS = {
    'ANPHY': 'C3-TP10',
    'DOD-H': 'C3-RPA',
    'Sleep-EDF-2013': 'Fpz-Cz',
    'Sleep-EDF-2018': 'Fpz-Cz',
}


def gather_subject_files(dataset_name, channel):
    data_root = os.path.join('dset', dataset_name, 'npz', channel)
    files = sorted(os.listdir(data_root))
    return [os.path.join(data_root, f) for f in files if f.endswith('.npz')]


def get_label_space(files):
    labels = set()
    for f in files:
        y = np.load(f)['y']
        labels.update(np.unique(y).astype(int).tolist())
    return sorted(labels)


def summarize_file(file_path, label_space):
    y = np.load(file_path)['y']
    total = int(y.shape[0])
    counts = {int(lbl): int(np.sum(y == lbl)) for lbl in label_space}
    dominant_label = max(counts, key=counts.get)
    dominant_share = counts[dominant_label] / total if total else 0.0
    return {
        'file': os.path.basename(file_path),
        'n_samples': total,
        'counts': counts,
        'dominant_label': dominant_label,
        'dominant_share': dominant_share,
        'label_entropy': float(-sum((c / total) * np.log2(c / total) for c in counts.values() if c > 0)) if total else 0.0,
    }


def summarize_dataset(dataset_name, channel):
    files = gather_subject_files(dataset_name, channel)
    label_space = get_label_space(files)
    subject_stats = [summarize_file(f, label_space) for f in files]

    total_counts = {lbl: 0 for lbl in label_space}
    for stats in subject_stats:
        for lbl, count in stats['counts'].items():
            total_counts[lbl] += count

    report = {
        'dataset': dataset_name,
        'channel': channel,
        'n_subjects': len(files),
        'label_space': label_space,
        'overall_counts': total_counts,
        'overall_distribution': {str(lbl): round(total_counts[lbl] / sum(total_counts.values()), 4) for lbl in label_space},
        'subject_stats': subject_stats,
        'mean_subject_samples': round(float(np.mean([s['n_samples'] for s in subject_stats])) if subject_stats else 0.0, 2),
        'std_subject_samples': round(float(np.std([s['n_samples'] for s in subject_stats])) if subject_stats else 0.0, 2),
        'min_subject_samples': min((s['n_samples'] for s in subject_stats), default=0),
        'max_subject_samples': max((s['n_samples'] for s in subject_stats), default=0),
    }
    return report


def summarize_folds(dataset_name):
    split_path = os.path.join('split_idx', f'idx_{dataset_name}.npy')
    splits = np.load(split_path, allow_pickle=True)
    files = gather_subject_files(dataset_name, DATASET_CHANNELS[dataset_name])
    label_space = get_label_space(files)

    fold_summaries = []
    for fold_idx, split in enumerate(splits, start=1):
        fold_summary = {'fold': fold_idx, 'train': {}, 'val': {}, 'test': {}}
        for set_name in ['train', 'val', 'test']:
            indices = split[set_name]
            counts = {lbl: 0 for lbl in label_space}
            for subject_idx in indices:
                subject_file = files[subject_idx]
                y = np.load(subject_file)['y']
                for lbl in label_space:
                    counts[lbl] += int(np.sum(y == lbl))
            fold_summary[set_name] = {
                'n_subjects': len(indices),
                'counts': counts,
                'distribution': {str(lbl): round(counts[lbl] / sum(counts.values()), 4) if sum(counts.values()) else 0.0 for lbl in label_space},
                'n_samples': int(sum(counts.values())),
            }
        fold_summaries.append(fold_summary)
    return fold_summaries


def make_label_count_matrix(dataset_name):
    files = gather_subject_files(dataset_name, DATASET_CHANNELS[dataset_name])
    labels = get_label_space(files)
    matrix = {str(lbl_a): {str(lbl_b): 0 for lbl_b in labels} for lbl_a in labels}
    for file_path in files:
        y = np.load(file_path)['y']
        for y_true in y:
            if int(y_true) in labels:
                matrix[str(int(y_true))][str(int(y_true))] += 1
    return matrix


def compare_datasets():
    results = {}
    for dataset_name in ['ANPHY', 'DOD-H']:
        results[dataset_name] = {
            'dataset': summarize_dataset(dataset_name, DATASET_CHANNELS[dataset_name]),
            'folds': summarize_folds(dataset_name),
            'label_matrix': make_label_count_matrix(dataset_name),
        }
    return results


def print_human_report(report):
    print('\n=== dataset comparison report ===')
    for dataset_name in ['ANPHY', 'DOD-H']:
        ds = report[dataset_name]['dataset']
        print(f'\n## {dataset_name}')
        print(f'  subjects: {ds["n_subjects"]}')
        print(f'  channel: {ds["channel"]}')
        print(f'  label space: {ds["label_space"]}')
        print(f'  overall counts: {ds["overall_counts"]}')
        print(f'  overall distribution: {ds["overall_distribution"]}')
        print(f'  mean subject length: {ds["mean_subject_samples"]} samples')
        print(f'  min/max subject length: {ds["min_subject_samples"]}/{ds["max_subject_samples"]}')

        best_subject = max(ds['subject_stats'], key=lambda s: s['dominant_share'])
        worst_subject = min(ds['subject_stats'], key=lambda s: s['dominant_share'])
        print(f'  subject with strongest majority: {best_subject["file"]} ({best_subject["dominant_label"]}, share={best_subject["dominant_share"]:.3f})')
        print(f'  subject with weakest majority: {worst_subject["file"]} ({worst_subject["dominant_label"]}, share={worst_subject["dominant_share"]:.3f})')

        print('  per-fold counts:')
        for fold in report[dataset_name]['folds']:
            print(f'    fold {fold["fold"]}: train={fold["train"]["counts"]}, val={fold["val"]["counts"]}, test={fold["test"]["counts"]}')

    print('\n=== quick comparison ===')
    anphy = report['ANPHY']['dataset']
    dod = report['DOD-H']['dataset']
    print('ANPHY vs DOD-H overall label proportions:')
    for lbl in sorted(set(anphy['label_space']) | set(dod['label_space'])):
        a_val = anphy['overall_distribution'].get(str(lbl), 0.0)
        d_val = dod['overall_distribution'].get(str(lbl), 0.0)
        print(f'  label {lbl}: ANPHY={a_val:.3f}, DOD-H={d_val:.3f}, delta={a_val - d_val:.3f}')


def save_report(report, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f'\nSaved JSON report to {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Analyze ANPHY and DOD-H sleep staging datasets for class balance and fold structure.')
    parser.add_argument('--out', type=str, default='results/dataset_diagnostics_report.json', help='output json path')
    args = parser.parse_args()

    report = compare_datasets()
    print_human_report(report)
    save_report(report, args.out)


if __name__ == '__main__':
    main()

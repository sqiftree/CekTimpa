"""Reproducible translation smoke benchmark, NOT a natural-image accuracy estimate.

Run: python -m experiments.run_features --samples 5 --output experiments/results/synthetic.csv
Each seed defines one source and its forged variant. No tuning on these results.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from src.detectors.copymove_block import detect_copymove_block
from src.metrics import calculate_pixel_metrics, calculate_image_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=Path('experiments/results/synthetic.csv'))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    rows = []
    for seed in range(args.seed, args.seed + args.samples):
        original = np.random.default_rng(seed).integers(0, 256, (128, 128, 3), dtype=np.uint8)
        forged = original.copy()
        forged[80:112, 80:112] = original[16:48, 16:48]
        gt = np.zeros((128, 128), dtype=np.uint8)
        gt[16:48, 16:48] = gt[80:112, 80:112] = 255
        for feature in ['A', 'B', 'C', 'D', 'H', 'DCT']:
            for label, image in [(0, original), (1, forged)]:
                config = dict(feature=feature, block_size=8, stride=2, eps=.01,
                              threshold_t=8, d_min=30, area_min=60)
                result = detect_copymove_block(image, config)
                metrics = calculate_pixel_metrics(result.mask, gt if label else np.zeros_like(gt))
                rows.append(dict(seed=seed, feature=feature, label=label,
                                 predicted=int(result.is_forged), **config_without_feature(config),
                                 **metrics, runtime_ms=result.timings['total_ms']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for feature in ['A', 'B', 'C', 'D', 'H', 'DCT']:
        subset = [r for r in rows if r['feature'] == feature]
        m = calculate_image_metrics([r['predicted'] for r in subset], [r['label'] for r in subset])
        f1 = np.mean([r['f1'] for r in subset if r['label']])
        print(f"{feature}: pixel F1={f1:.3f}, image recall={m['recall']:.3f}, FPR={m['fpr']:.3f}")
    print(f'Results: {args.output}. Synthetic translation only; epsilon is not calibrated per feature.')


def config_without_feature(config):
    return {k: v for k, v in config.items() if k != 'feature'}


if __name__ == '__main__':
    main()

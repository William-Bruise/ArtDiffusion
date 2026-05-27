import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--log', default='outputs/default/logs/train_loss.jsonl')
    p.add_argument('--out', default='outputs/default/logs/train_loss_logscale.png')
    args = p.parse_args()

    path = Path(args.log)
    if not path.exists():
        raise FileNotFoundError(f'Loss log not found: {path}')

    steps, losses = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            steps.append(row['step'])
            losses.append(max(float(row['loss']), 1e-12))

    if not steps:
        raise RuntimeError('No loss records found in log file')

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, losses, linewidth=1.4)
    plt.yscale('log')
    plt.xlabel('Step')
    plt.ylabel('Training Loss (log scale)')
    plt.title('Diffusion Training Loss Curve')
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f'Saved: {args.out}')

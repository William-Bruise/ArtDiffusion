import argparse

from datasets.prepare import prepare_dataset


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data')
    p.add_argument('--dataset', choices=['ffhq', 'celebahq'], required=True)
    p.add_argument('--train_split', type=float, default=0.98)
    args = p.parse_args()
    out = prepare_dataset(args.root, args.dataset, args.train_split)
    print(out)

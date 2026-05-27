import argparse

from _bootstrap import bootstrap_repo_root

bootstrap_repo_root()

from trainers.trainer import Trainer
from utils import load_config

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/base.yaml')
    args = p.parse_args()
    cfg = load_config(args.config)
    Trainer(cfg).train()

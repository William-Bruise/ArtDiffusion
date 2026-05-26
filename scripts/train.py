from trainers.trainer import Trainer
from utils import load_config
import argparse

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/base.yaml')
    args = p.parse_args()
    cfg = load_config(args.config)
    Trainer(cfg).train()

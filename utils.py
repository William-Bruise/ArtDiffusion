import os
import random
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


def load_checkpoint(path: str, map_location='cpu') -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)

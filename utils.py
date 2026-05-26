import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _auto_number(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        # support scientific notation strings like "2e-4"
        try:
            if any(ch in s.lower() for ch in ["e", "."]):
                return float(s)
            return int(s)
        except ValueError:
            return v
    if isinstance(v, dict):
        return {k: _auto_number(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_auto_number(x) for x in v]
    return v


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return _auto_number(cfg)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


def load_checkpoint(path: str, map_location='cpu') -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)

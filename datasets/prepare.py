import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List

import requests
from tqdm import tqdm

DATA_SOURCES = {
    'ffhq': [
        # metadata/public thumbnails fallback; full FFHQ often needs external tools or kaggle auth
        'https://github.com/NVlabs/ffhq-dataset/raw/master/README.md',
    ],
    'celebahq': [
        'https://github.com/switchablenorms/CelebAMask-HQ/raw/master/CelebA-HQ-img.zip',
    ],
}


def _download(url: str, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        resume_header = {}
        mode = 'wb'
        if dst.exists():
            downloaded = dst.stat().st_size
            resume_header = {'Range': f'bytes={downloaded}-'}
            mode = 'ab'
        else:
            downloaded = 0
        with requests.get(url, stream=True, timeout=30, headers=resume_header) as r:
            if r.status_code not in (200, 206):
                return False
            total = int(r.headers.get('Content-Length', 0)) + downloaded
            with open(dst, mode) as f, tqdm(total=total, initial=downloaded, unit='B', unit_scale=True, desc=dst.name) as pbar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception:
        return False


def _extract_if_needed(archive: Path, out_dir: Path) -> None:
    if archive.suffix == '.zip':
        with zipfile.ZipFile(archive, 'r') as zf:
            zf.extractall(out_dir)


def _collect_images(root: Path) -> List[str]:
    exts = {'.jpg', '.jpeg', '.png', '.webp'}
    return sorted([str(p.relative_to(root)) for p in root.rglob('*') if p.suffix.lower() in exts])


def _split_and_index(images: List[str], train_split: float, out_path: Path) -> None:
    n_train = int(len(images) * train_split)
    index = {'train': images[:n_train], 'val': images[n_train:]}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2)


def prepare_dataset(root: str, dataset: str, train_split: float = 0.98) -> Dict[str, str]:
    root_p = Path(root)
    ds_dir = root_p / dataset
    img_dir = ds_dir / 'images'
    index_path = ds_dir / 'index.json'
    ds_dir.mkdir(parents=True, exist_ok=True)

    if index_path.exists() and img_dir.exists() and any(img_dir.rglob('*')):
        return {'status': 'ready', 'index': str(index_path)}

    img_dir.mkdir(parents=True, exist_ok=True)

    source_list = DATA_SOURCES.get(dataset, [])
    downloaded_any = False
    for i, url in enumerate(source_list):
        fname = ds_dir / f'source_{i}{Path(url).suffix or ".bin"}'
        ok = _download(url, fname)
        if not ok:
            continue
        downloaded_any = True
        if fname.suffix == '.zip':
            _extract_if_needed(fname, img_dir)

    images = _collect_images(img_dir)

    if len(images) == 0:
        raise RuntimeError(
            f'Failed to auto-prepare {dataset}. '\
            f'Please manually place images under {img_dir} and rerun. '\
            'Some official sources require authentication/license acceptance (common for FFHQ/CelebA-HQ).'
        )

    _split_and_index(images, train_split, index_path)
    return {'status': 'prepared' if downloaded_any else 'manual_ready', 'index': str(index_path)}

import argparse
import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from _bootstrap import bootstrap_repo_root

bootstrap_repo_root()


CANDIDATE_COLUMNS = ["image", "img", "jpg", "png", "bytes"]


def infer_image_column(sample: dict) -> str:
    for c in CANDIDATE_COLUMNS:
        if c in sample:
            return c
    for k, v in sample.items():
        if isinstance(v, (bytes, bytearray)):
            return k
        if isinstance(v, dict) and ("bytes" in v or "path" in v):
            return k
    raise RuntimeError(f"Unable to infer image column from keys: {list(sample.keys())}")


def decode_image(value):
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    raise ValueError(f"Unsupported image value type: {type(value)}")


def main():
    p = argparse.ArgumentParser(description="Convert FFHQ parquet shards to images.")
    p.add_argument("--parquet_dir", required=True)
    p.add_argument("--out_dir", default="data/ffhq/images")
    p.add_argument("--ext", default="jpg", choices=["jpg", "png"])
    p.add_argument("--quality", type=int, default=95)
    args = p.parse_args()

    parquet_dir = Path(args.parquet_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found in {parquet_dir}")

    first_table = pq.read_table(files[0], columns=None)
    first_sample = first_table.slice(0, 1).to_pylist()[0]
    img_col = infer_image_column(first_sample)
    print(f"Detected image column: {img_col}")

    idx = 0
    for pf in files:
        table = pq.read_table(pf, columns=None)
        rows = table.to_pylist()
        for row in tqdm(rows, desc=pf.name):
            img = decode_image(row[img_col])
            out_path = out_dir / f"{idx:06d}.{args.ext}"
            if args.ext == "jpg":
                img.save(out_path, quality=args.quality)
            else:
                img.save(out_path)
            idx += 1

    print(f"Done. Exported {idx} images to {out_dir}")


if __name__ == "__main__":
    main()

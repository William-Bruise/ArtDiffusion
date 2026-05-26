# ArtDiffusion: Continuous Random Field Image Generation

研究型最小可运行仓库：学习连续坐标域上的无条件图像随机函数，而非固定分辨率生成器。

## Features
- Continuous coordinate-set DDPM objective (random coordinate subsets).
- Global context + coordinate-conditional local denoiser.
- Regular / irregular coordinate sampling.
- Cross-subset consistency loss.
- Variable resolution/aspect ratio sampling.
- Unified low-level inverse operators + posterior inference demo.
- Dataset preparation with auto-download attempts + robust fallback instructions.

## Structure
- `configs/`
- `datasets/`
- `models/`
- `trainers/`
- `samplers/`
- `inverse_problems/`
- `metrics/`
- `scripts/`
- `docs/`

## Quickstart
```bash
pip install -r requirements.txt
python scripts/prepare_data.py --dataset celebahq --root data
python scripts/prepare_data.py --dataset ffhq --root data
python scripts/train.py --config configs/base.yaml
python scripts/sample.py --ckpt outputs/default/ckpts/2000.pt --h 256 --w 384 --n 4
python scripts/inverse_demo.py --ckpt outputs/default/ckpts/2000.pt --task inpainting --input your_image.png
```

> 若自动下载失败，请按报错信息手动放置图片到 `data/<dataset>/images/` 后重跑准备脚本。

## Math and Bayesian details
See `docs/method.md`.

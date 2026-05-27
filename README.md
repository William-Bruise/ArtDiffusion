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
python scripts/inverse_demo.py --ckpt outputs/default/ckpts/2000.pt --task inpainting --input your_image.png
```

## Sampling（重点：怎么运行采样）
先确认你已经有 checkpoint（例如 `outputs/default/ckpts/2000.pt`），然后在仓库根目录执行。

### 1) 最常用命令（任意分辨率 + 任意长宽比）
```bash
python scripts/sample.py \
  --ckpt outputs/default/ckpts/2000.pt \
  --h 256 --w 384 \
  --n 4 \
  --steps 50 \
  --seed 42 \
  --out outputs/samples_256x384_seed42.png
```

参数说明：
- `--ckpt`：训练得到的 checkpoint 路径（必填）
- `--h --w`：输出高度/宽度（可任意设置）
- `--n`：一次采样张数
- `--steps`：反向采样步数（更大通常更慢但更稳）
- `--seed`：随机种子（用于复现实验）
- `--out`：输出图片文件路径

### 2) 不同分辨率对比（同一个 seed）
```bash
python scripts/sample.py --ckpt outputs/default/ckpts/2000.pt --h 128 --w 128 --n 1 --seed 123 --out outputs/sample_128.png
python scripts/sample.py --ckpt outputs/default/ckpts/2000.pt --h 256 --w 256 --n 1 --seed 123 --out outputs/sample_256.png
python scripts/sample.py --ckpt outputs/default/ckpts/2000.pt --h 256 --w 384 --n 1 --seed 123 --out outputs/sample_256x384.png
```

### 3) 采样命令报错时优先检查
```bash
ls outputs/default/ckpts
python -m compileall scripts/sample.py samplers/ddim_sampler.py models/continuous_diffusion.py
```
- 若 `--ckpt` 文件不存在，先训练或改为真实 checkpoint 路径。
- 若显存不足，先把 `--h/--w` 或 `--n` 调小。

### 4) 训练到 2~20 万步仍然很噪怎么办
- 请使用**最新代码重新训练**（旧 checkpoint 学到的是旧目标函数，视觉上可能长期停在噪声态）。
- 当前版本已将局部特征图（CNN）注入到坐标去噪器，旧 checkpoint 不兼容新结构，必须从头训练。
- 已修复一个关键一致性问题：训练时局部特征现在从 `x_t`（噪声状态）提取，以匹配采样阶段输入分布。若你是在修复前训练的，请重新训练。
- 建议先用 `--steps 100` 采样再看质量，50 步对当前小模型常偏噪。
- 关注训练日志中的 `fine/coarse/cons` 是否同步下降；若 `cons` 不降，优先减小 `coordinates.consistency_weight` 到 `0.1`。

### 5) GPU 是否生效（你问的这个）
训练启动后会打印设备信息：
- `device=cuda gpu=... amp=True` 代表已经使用 GPU。
- `device=cpu` 代表当前环境没有可用 CUDA（常见原因：PyTorch 装的是 CPU 版本，或驱动/CUDA 不匹配）。

可用以下命令快速自检：
```bash
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('torch_cuda=', torch.version.cuda); print('device_count=', torch.cuda.device_count())"
```

> 若自动下载失败，请按报错信息手动放置图片到 `data/<dataset>/images/` 后重跑准备脚本。


### FFHQ parquet 导出（HuggingFace 分片）
如果你下载的是 `*.parquet` 分片（不是 zip），请执行：
```bash
python scripts/convert_ffhq_parquet.py --parquet_dir /home/wuweihao/Datasets/FFHQ --out_dir data/ffhq/images
python scripts/prepare_data.py --dataset ffhq --root data
```

> 注意：不要在仓库根目录运行 `from datasets import load_dataset`，因为本仓库有 `datasets/` 本地包同名，会发生模块名冲突。

## Math and Bayesian details
See `docs/method.md`.

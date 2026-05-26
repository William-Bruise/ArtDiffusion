import argparse
import torch
from torchvision.io import read_image
from torchvision.utils import save_image

from inverse_problems.operators import DenoiseOperator, InpaintingOperator, SuperResolutionOperator
from inverse_problems.solver import posterior_sample
from models.continuous_diffusion import ContinuousFieldDiffusion
from utils import load_checkpoint

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--task', choices=['inpainting', 'superres', 'denoise'], required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--out', default='outputs/inverse.png')
    args = p.parse_args()
    ckpt = load_checkpoint(args.ckpt)
    model = ContinuousFieldDiffusion(**ckpt['cfg']['model'])
    model.load_state_dict(ckpt['model'])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(dev).eval()

    x = read_image(args.input).float().unsqueeze(0).to(dev) / 127.5 - 1
    if args.task == 'inpainting':
        mask = torch.ones_like(x)
        mask[:, :, :, x.shape[-1] // 4: 3 * x.shape[-1] // 4] = 0
        op = InpaintingOperator(mask)
    elif args.task == 'superres':
        op = SuperResolutionOperator(scale=4)
    else:
        op = DenoiseOperator(sigma=0.1)

    y = op.forward(x)
    rec = posterior_sample(model, y, op, resolution=(x.shape[-2], x.shape[-1]))
    save_image((rec.clamp(-1, 1) + 1) / 2, args.out)

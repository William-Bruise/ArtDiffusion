import argparse
import torch
from torchvision.utils import save_image

from models.continuous_diffusion import ContinuousFieldDiffusion
from samplers.ddim_sampler import sample_grid
from utils import load_checkpoint

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--h', type=int, default=128)
    p.add_argument('--w', type=int, default=128)
    p.add_argument('--n', type=int, default=4)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='outputs/samples.png')
    args = p.parse_args()
    ckpt = load_checkpoint(args.ckpt)
    model = ContinuousFieldDiffusion(**ckpt['cfg']['model'])
    model.load_state_dict(ckpt['model'])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(dev).eval()
    with torch.no_grad():
        x = sample_grid(model, dev, args.n, (args.h, args.w), args.steps, args.seed)
    save_image((x.clamp(-1, 1) + 1) / 2, args.out, nrow=min(4, args.n))

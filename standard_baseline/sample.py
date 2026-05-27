import argparse
from pathlib import Path
import sys
import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from standard_baseline.model import UNetSmall
from standard_baseline.ddpm import make_schedule


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--h', type=int, default=128)
    p.add_argument('--w', type=int, default=128)
    p.add_argument('--n', type=int, default=4)
    p.add_argument('--out', default='outputs/standard_baseline/samples.png')
    args = p.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNetSmall().to(dev)
    state = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(state['model'])
    model.eval()

    betas, alphas, abar = make_schedule(device=dev)
    x = torch.randn(args.n, 3, args.h, args.w, device=dev)
    T = len(betas)
    with torch.no_grad():
        for i in reversed(range(T)):
            t = torch.full((args.n,), i / (T - 1), device=dev)
            eps = model(x, t)
            a = alphas[i]
            ab = abar[i]
            coef1 = 1 / a.sqrt()
            coef2 = (1 - a) / (1 - ab).sqrt()
            mean = coef1 * (x - coef2 * eps)
            if i > 0:
                z = torch.randn_like(x)
                sigma = betas[i].sqrt()
                x = mean + sigma * z
            else:
                x = mean
    save_image((x.clamp(-1, 1) + 1) / 2, args.out, nrow=min(args.n, 4))


if __name__ == '__main__':
    main()

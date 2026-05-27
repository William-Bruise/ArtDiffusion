import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from scripts._bootstrap import bootstrap_repo_root
bootstrap_repo_root()
from datasets.image_field_dataset import ImageFieldDataset
from standard_baseline.model import UNetSmall
from standard_baseline.ddpm import make_schedule, q_sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data')
    p.add_argument('--dataset', default='ffhq')
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--steps', type=int, default=200000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--out', default='outputs/standard_baseline')
    args = p.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = ImageFieldDataset(args.root, args.dataset, 'train', args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    model = UNetSmall().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    betas, alphas, abar = make_schedule(device=dev)
    out = Path(args.out)
    (out / 'ckpts').mkdir(parents=True, exist_ok=True)
    (out / 'samples').mkdir(parents=True, exist_ok=True)

    step = 0
    while step < args.steps:
        for x0 in dl:
            x0 = x0.to(dev)
            b = x0.shape[0]
            t = torch.randint(0, len(abar), (b,), device=dev)
            eps = torch.randn_like(x0)
            xt = q_sample(x0, t, abar, eps)
            t_norm = t.float() / (len(abar) - 1)
            pred = model(xt, t_norm)
            loss = torch.mean((pred - eps) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 100 == 0:
                print(f'step={step} loss={loss.item():.6f}')
            if step % 2000 == 0:
                torch.save({'model': model.state_dict(), 'step': step}, out / 'ckpts' / f'{step}.pt')
            if step >= args.steps:
                break


if __name__ == '__main__':
    main()

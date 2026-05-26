import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from datasets.image_field_dataset import ImageFieldDataset
from models.continuous_diffusion import ContinuousFieldDiffusion, q_sample
from samplers.ddim_sampler import sample_grid
from trainers.coordinate_sampler import bilinear_sample, sample_coords
from utils import ensure_dir, load_checkpoint, save_checkpoint, set_seed


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        set_seed(cfg['experiment']['seed'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.out = Path(cfg['experiment']['out_dir'])
        ensure_dir(str(self.out / 'ckpts'))
        ensure_dir(str(self.out / 'samples'))

        ds_name = cfg['data']['dataset']
        if ds_name == 'mixed':
            raise NotImplementedError('Use provided single dataset configs for training; mixed can be built by concatenating datasets.')
        ds = ImageFieldDataset(cfg['data']['root'], ds_name, 'train', cfg['data']['image_size'])
        self.loader = DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'], drop_last=True)

        self.model = ContinuousFieldDiffusion(**cfg['model']).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
        self.step = 0

    def maybe_resume(self):
        path = self.cfg['train']['resume']
        if path:
            ckpt = load_checkpoint(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            self.opt.load_state_dict(ckpt['opt'])
            self.step = ckpt['step']

    def train(self):
        self.maybe_resume()
        max_steps = self.cfg['train']['max_steps']
        c = self.cfg['coordinates']
        pbar = tqdm(total=max_steps, initial=self.step)
        while self.step < max_steps:
            for img in self.loader:
                img = img.to(self.device)
                b, _, h, w = img.shape
                coords_f = sample_coords(b, h, w, c['n_points_min'], c['n_points_max'], c['irregular_prob'], self.device)
                n_f = coords_f.shape[1]
                perm = torch.randperm(n_f, device=self.device)
                n_c = max(16, int(c['coarse_ratio'] * n_f))
                coords_c = coords_f[:, perm[:n_c], :]

                x0_f = bilinear_sample(img, coords_f)
                x0_c = bilinear_sample(img, coords_c)
                t = torch.rand(b, device=self.device)
                eps_f = torch.randn_like(x0_f)
                eps_c = torch.randn_like(x0_c)
                xt_f, _, _ = q_sample(x0_f, t, eps_f)
                xt_c, _, _ = q_sample(x0_c, t, eps_c)

                g = self.model.encode_global(img)
                pred_f = self.model(xt_f, coords_f, t, g)
                pred_c = self.model(xt_c, coords_c, t, g)

                loss_main = F.mse_loss(pred_f, eps_f)
                # subset consistency: evaluate coarse coords from fine context noising level
                xt_c_from_f = bilinear_sample(img, coords_c)
                pred_c_from_f = self.model(xt_c_from_f, coords_c, t, g)
                loss_cons = F.mse_loss(pred_c, pred_c_from_f)
                loss = loss_main + c['consistency_weight'] * loss_cons

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['train']['grad_clip'])
                self.opt.step()

                self.step += 1
                pbar.update(1)
                if self.step % self.cfg['train']['log_every'] == 0:
                    pbar.set_description(f'loss={loss.item():.4f}')
                if self.step % self.cfg['train']['save_every'] == 0:
                    save_checkpoint({'model': self.model.state_dict(), 'opt': self.opt.state_dict(), 'step': self.step, 'cfg': self.cfg}, str(self.out / 'ckpts' / f'{self.step}.pt'))
                if self.step % self.cfg['train']['sample_every'] == 0:
                    smp = sample_grid(self.model, self.device, 1, (h, w), self.cfg['sampling']['num_steps'])
                    save_image((smp.clamp(-1, 1) + 1) / 2, self.out / 'samples' / f'{self.step}.png')
                if self.step >= max_steps:
                    break

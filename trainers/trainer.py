import os
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from datasets.image_field_dataset import ImageFieldDataset
from models.continuous_diffusion import ContinuousFieldDiffusion, cosine_alpha_bar
from samplers.ddim_sampler import sample_grid
from trainers.coordinate_sampler import bilinear_sample, sample_coords
from utils import ensure_dir, load_checkpoint, save_checkpoint, set_seed


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        set_seed(cfg['experiment']['seed'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_amp = bool(cfg['train'].get('use_amp', True) and self.device == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.out = Path(cfg['experiment']['out_dir'])
        ensure_dir(str(self.out / 'ckpts'))
        ensure_dir(str(self.out / 'samples'))
        ensure_dir(str(self.out / 'logs'))
        self.loss_log_path = self.out / 'logs' / 'train_loss.jsonl'
        self.loss_plot_path = self.out / 'logs' / 'train_loss_logscale.png'
        self.loss_history_steps = []
        self.loss_history_vals = []

        ds_name = cfg['data']['dataset']
        if ds_name == 'mixed':
            raise NotImplementedError('Use provided single dataset configs for training; mixed can be built by concatenating datasets.')
        ds = ImageFieldDataset(cfg['data']['root'], ds_name, 'train', cfg['data']['image_size'])
        self.loader = DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'], drop_last=True)

        self.model = ContinuousFieldDiffusion(**cfg['model']).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
        self.step = 0
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            print(f'[Trainer] device=cuda gpu="{gpu_name}" amp={self.use_amp}')
        else:
            print('[Trainer] device=cpu (CUDA not available in current PyTorch/runtime)')

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

                t = torch.rand(b, device=self.device)

                use_global_encoder = bool(self.cfg['train'].get('use_global_encoder', False))
                if use_global_encoder:
                    mu, logvar = self.model.encode_global_stats(img)
                    g = self.model.sample_global_latent(mu, logvar)
                else:
                    mu = logvar = None
                    g = torch.randn(b, self.model.global_latent_dim, device=self.device)
                # Build a single noisy image-space state x_t and derive all subset targets from it.
                # This guarantees target eps at coordinates matches the conditioning xt distribution.
                eps_img = torch.randn_like(img)
                ab_img = cosine_alpha_bar(t)[:, None, None, None]
                xt_img = torch.sqrt(ab_img) * img + torch.sqrt(1 - ab_img) * eps_img
                xt_f = bilinear_sample(xt_img, coords_f)
                xt_c = bilinear_sample(xt_img, coords_c)
                eps_f = bilinear_sample(eps_img, coords_f)
                eps_c = bilinear_sample(eps_img, coords_c)
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    pred_eps_img = self.model.denoise_grid(xt_img, t, g)
                    pred_f = bilinear_sample(pred_eps_img, coords_f)
                    pred_c = bilinear_sample(pred_eps_img, coords_c)

                    loss_full = F.mse_loss(pred_eps_img, eps_img)
                    objective_mode = self.cfg['train'].get('objective_mode', 'full_eps_mse')
                    if objective_mode != 'full_eps_mse':
                        raise ValueError("Only train.objective_mode=full_eps_mse is supported in the standardized training path")
                    loss = self.cfg['train'].get('full_weight', 1.0) * loss_full

                self.opt.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['train']['grad_clip'])
                self.scaler.step(self.opt)
                self.scaler.update()

                self.step += 1
                pbar.update(1)
                if self.step % self.cfg['train']['log_every'] == 0:
                    pbar.set_description(f'loss={loss.item():.6f} mode={objective_mode}')
                    self.loss_history_steps.append(self.step)
                    self.loss_history_vals.append(max(float(loss.item()), 1e-12))
                    with open(self.loss_log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({'step': self.step, 'loss': float(loss.item()), 'mode': objective_mode}) + '\n')
                    self._save_loss_plot()
                if self.step % self.cfg['train']['save_every'] == 0:
                    save_checkpoint({'model': self.model.state_dict(), 'opt': self.opt.state_dict(), 'step': self.step, 'cfg': self.cfg}, str(self.out / 'ckpts' / f'{self.step}.pt'))
                if self.step % self.cfg['train']['sample_every'] == 0:
                    self.model.eval()
                    with torch.no_grad():
                        smp = sample_grid(self.model, self.device, 1, (h, w), self.cfg['sampling']['num_steps'])
                    self.model.train()
                    save_image((smp.clamp(-1, 1) + 1) / 2, self.out / 'samples' / f'{self.step}.png')
                if self.step >= max_steps:
                    break

    def _save_loss_plot(self):
        if not self.loss_history_steps:
            return
        plt.figure(figsize=(8, 4.5))
        plt.plot(self.loss_history_steps, self.loss_history_vals, linewidth=1.4)
        plt.yscale('log')
        plt.xlabel('Step')
        plt.ylabel('Training Loss (log scale)')
        plt.title('Diffusion Training Loss Curve')
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(self.loss_plot_path, dpi=160)
        plt.close()

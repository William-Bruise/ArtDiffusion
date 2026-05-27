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
        self.use_amp = bool(cfg['train'].get('use_amp', True) and self.device == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
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

                x0_f = bilinear_sample(img, coords_f)
                x0_c = bilinear_sample(img, coords_c)
                t = torch.rand(b, device=self.device)
                eps_f = torch.randn_like(x0_f)
                xt_f, _, _ = q_sample(x0_f, t, eps_f)
                eps_c = torch.randn_like(x0_c)
                xt_c, _, _ = q_sample(x0_c, t, eps_c)

                mu, logvar = self.model.encode_global_stats(img)
                g = self.model.sample_global_latent(mu, logvar)
                # Keep train/inference consistent: local features must be extracted from current noisy state x_t,
                # not from clean x0 image.
                xt_f_img = xt_f.reshape(b, h, w, 3).permute(0, 3, 1, 2)
                xt_c_img = xt_c.reshape(b, h, w, 3).permute(0, 3, 1, 2)
                feat_map_f = self.model.encode_local_map(xt_f_img)
                feat_map_c = self.model.encode_local_map(xt_c_img)
                lf_f = self.model.sample_local_features(feat_map_f, coords_f)
                lf_c = self.model.sample_local_features(feat_map_c, coords_c)
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    pred_f = self.model(xt_f, coords_f, t, g, lf_f)
                    pred_c = self.model(xt_c, coords_c, t, g, lf_c)

                    loss_main_f = F.mse_loss(pred_f, eps_f)
                    loss_main_c = F.mse_loss(pred_c, eps_c)
                    # subset consistency: same noise realization restricted to coarse subset
                    eps_f_on_c = eps_f[:, perm[:n_c], :]
                    xt_c_from_f, _, _ = q_sample(x0_c, t, eps_f_on_c)
                    xt_c_from_f_img = xt_c_from_f.reshape(b, h, w, 3).permute(0, 3, 1, 2)
                    feat_map_c_from_f = self.model.encode_local_map(xt_c_from_f_img)
                    lf_c_from_f = self.model.sample_local_features(feat_map_c_from_f, coords_c)
                    pred_c_from_f = self.model(xt_c_from_f, coords_c, t, g, lf_c_from_f)
                    # consistency should be anchored to a target noise, not only prediction-vs-prediction
                    loss_cons = F.mse_loss(pred_c_from_f, eps_f_on_c)
                    loss_kl = self.model.kl_global_prior(mu, logvar)
                    loss = (
                        loss_main_f
                        + c.get('coarse_weight', 0.5) * loss_main_c
                        + c['consistency_weight'] * loss_cons
                        + self.cfg['train'].get('kl_weight', 1e-4) * loss_kl
                    )

                self.opt.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['train']['grad_clip'])
                self.scaler.step(self.opt)
                self.scaler.update()

                self.step += 1
                pbar.update(1)
                if self.step % self.cfg['train']['log_every'] == 0:
                    pbar.set_description(
                        f'loss={loss.item():.4f} fine={loss_main_f.item():.4f} coarse={loss_main_c.item():.4f} cons={loss_cons.item():.4f} kl={loss_kl.item():.4f}'
                    )
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

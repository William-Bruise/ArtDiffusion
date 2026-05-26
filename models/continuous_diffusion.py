from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_encode(coords: torch.Tensor, n_freq: int = 32) -> torch.Tensor:
    freqs = 2 ** torch.linspace(0, n_freq - 1, n_freq, device=coords.device)
    x = coords[..., None] * freqs
    return torch.cat([torch.sin(2 * torch.pi * x), torch.cos(2 * torch.pi * x)], dim=-1).flatten(-2)


class CoordMLP(nn.Module):
    def __init__(self, in_dim: int, hid: int, layers: int, out_dim: int = 3, dropout: float = 0.0):
        super().__init__()
        net = []
        d = in_dim
        for _ in range(layers):
            net += [nn.Linear(d, hid), nn.SiLU(), nn.Dropout(dropout)]
            d = hid
        net += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ContinuousFieldDiffusion(nn.Module):
    def __init__(self, time_embed_dim=128, global_latent_dim=256, coord_hidden_dim=256, coord_fourier_dim=64, num_layers=6, dropout=0.1):
        super().__init__()
        self.coord_fourier_dim = coord_fourier_dim
        coord_dim = coord_fourier_dim * 4
        self.global_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(128, global_latent_dim, 4, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.t_embed = nn.Sequential(nn.Linear(1, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim))
        self.coord_mlp = CoordMLP(coord_dim + global_latent_dim + time_embed_dim + 3, coord_hidden_dim, num_layers, 3, dropout)

    def encode_global(self, x_img: torch.Tensor) -> torch.Tensor:
        return self.global_encoder(x_img).flatten(1)

    def forward(self, noisy_rgb: torch.Tensor, coords: torch.Tensor, t: torch.Tensor, global_context: torch.Tensor) -> torch.Tensor:
        # noisy_rgb: [B,N,3], coords: [B,N,2], t:[B]
        c = fourier_encode(coords, self.coord_fourier_dim)
        te = self.t_embed(t[:, None]).unsqueeze(1).expand(-1, coords.shape[1], -1)
        g = global_context.unsqueeze(1).expand(-1, coords.shape[1], -1)
        inp = torch.cat([noisy_rgb, c, te, g], dim=-1)
        return self.coord_mlp(inp)


def cosine_alpha_bar(t):
    return torch.cos((t + 0.008) / 1.008 * torch.pi / 2) ** 2


def q_sample(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ab = cosine_alpha_bar(t)[:, None, None]
    xt = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps
    return xt, ab, 1 - ab

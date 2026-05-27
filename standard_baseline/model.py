import math
import torch
import torch.nn as nn


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.emb = nn.Linear(t_dim, c_out)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.conv1(x))
        h = h + self.emb(t_emb)[:, :, None, None]
        h = self.act(self.conv2(h))
        return h + self.skip(x)


class UNetSmall(nn.Module):
    def __init__(self, base=64, t_dim=256):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.in_conv = nn.Conv2d(3, base, 3, padding=1)
        self.down1 = ResBlock(base, base, t_dim)          # H
        self.down2 = ResBlock(base, base * 2, t_dim)      # H/2
        self.pool = nn.AvgPool2d(2)
        self.mid = ResBlock(base * 2, base * 2, t_dim)    # H/4
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1 = ResBlock(base * 2 + base * 2, base * 2, t_dim)  # H/2
        self.up2 = ResBlock(base * 2 + base, base, t_dim)           # H
        self.out = nn.Conv2d(base, 3, 3, padding=1)
        self.t_dim = t_dim

    def forward(self, x, t):
        t_emb = self.t_mlp(timestep_embedding(t, self.t_dim))
        x0 = self.in_conv(x)
        d1 = self.down1(x0, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        m = self.mid(self.pool(d2), t_emb)
        u = self.up(m)
        u = self.up1(torch.cat([u, d2], dim=1), t_emb)
        u = self.up(u)
        u = self.up2(torch.cat([u, d1], dim=1), t_emb)
        return self.out(u)

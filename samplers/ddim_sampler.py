import torch

from models.continuous_diffusion import cosine_alpha_bar


def sample_grid(model, device, batch, resolution=(128, 128), steps=50, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    h, w = resolution
    ys, xs = torch.meshgrid(torch.linspace(0, 1, h, device=device), torch.linspace(0, 1, w, device=device), indexing='ij')
    coords = torch.stack([xs, ys], dim=-1).reshape(1, -1, 2).repeat(batch, 1, 1)
    x = torch.randn(batch, h * w, 3, device=device)
    g = torch.randn(batch, model.global_latent_dim, device=device)
    times = torch.linspace(1, 1e-3, steps, device=device)
    for i in range(steps - 1):
        t = times[i].repeat(batch)
        tn = times[i + 1].repeat(batch)
        ab_t = cosine_alpha_bar(t)[:, None, None]
        ab_n = cosine_alpha_bar(tn)[:, None, None]
        eps = model(x, coords, t, g)
        x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp_min(1e-6)
        x = ab_n.sqrt() * x0 + (1 - ab_n).sqrt() * eps
    img = x.reshape(batch, h, w, 3).permute(0, 3, 1, 2)
    return img

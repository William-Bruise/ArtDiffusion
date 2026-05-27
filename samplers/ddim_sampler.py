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
    # cosine schedule has alpha_bar(1)=0 exactly; avoid singular x0 reconstruction at t=1
    times = torch.linspace(0.999, 1e-3, steps, device=device)
    for i in range(steps - 1):
        t = times[i].repeat(batch)
        tn = times[i + 1].repeat(batch)
        ab_t = cosine_alpha_bar(t)[:, None, None]
        ab_n = cosine_alpha_bar(tn)[:, None, None]
        x_img = x.reshape(batch, h, w, 3).permute(0, 3, 1, 2)
        eps_img = model.denoise_grid(x_img, t, g)
        eps = eps_img.permute(0, 2, 3, 1).reshape(batch, h * w, 3)
        x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp_min(1e-4)
        x0 = x0.clamp(-1, 1)
        x = ab_n.sqrt() * x0 + (1 - ab_n).sqrt() * eps
    # final denoise to x0 at the last time step for cleaner samples
    t_last = times[-1].repeat(batch)
    ab_last = cosine_alpha_bar(t_last)[:, None, None]
    x_img = x.reshape(batch, h, w, 3).permute(0, 3, 1, 2)
    eps_last_img = model.denoise_grid(x_img, t_last, g)
    eps_last = eps_last_img.permute(0, 2, 3, 1).reshape(batch, h * w, 3)
    x = ((x - (1 - ab_last).sqrt() * eps_last) / ab_last.sqrt().clamp_min(1e-4)).clamp(-1, 1)
    img = x.reshape(batch, h, w, 3).permute(0, 3, 1, 2)
    return img

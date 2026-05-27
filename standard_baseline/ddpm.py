import torch


def make_schedule(T=1000, beta1=1e-4, beta2=0.02, device='cpu'):
    betas = torch.linspace(beta1, beta2, T, device=device)
    alphas = 1 - betas
    abar = torch.cumprod(alphas, dim=0)
    return betas, alphas, abar


def q_sample(x0, t_idx, abar, noise):
    a = abar[t_idx][:, None, None, None]
    return a.sqrt() * x0 + (1 - a).sqrt() * noise

import torch
import torch.nn.functional as F

from samplers.ddim_sampler import sample_grid


def posterior_sample(model, y, operator, steps=200, lr=1e-1, resolution=(128, 128), seed=0):
    device = y.device
    x = sample_grid(model, device, y.shape[0], resolution=resolution, steps=30, seed=seed).detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        pred = operator.forward(x)
        data_loss = F.mse_loss(pred, y)
        prior_reg = (x ** 2).mean() * 1e-4
        loss = data_loss + prior_reg
        opt.zero_grad()
        loss.backward()
        opt.step()
        x.data.clamp_(-1, 1)
    return x.detach()

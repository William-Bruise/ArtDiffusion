import torch


def sample_coords(batch: int, h: int, w: int, n_min: int, n_max: int, irregular_prob: float, device: str):
    n = torch.randint(n_min, n_max + 1, (1,)).item()
    if torch.rand(1).item() < irregular_prob:
        coords = torch.rand(batch, n, 2, device=device)
    else:
        side = int(n ** 0.5)
        ys, xs = torch.meshgrid(torch.linspace(0, 1, side, device=device), torch.linspace(0, 1, side, device=device), indexing='ij')
        g = torch.stack([xs, ys], dim=-1).reshape(1, -1, 2)
        coords = g.repeat(batch, 1, 1)
    return coords


def bilinear_sample(img: torch.Tensor, coords: torch.Tensor):
    b, _, h, w = img.shape
    grid = coords.clone()
    grid = grid * 2 - 1
    grid = grid.view(b, 1, -1, 2)
    sampled = torch.nn.functional.grid_sample(img, grid, mode='bilinear', align_corners=True)
    return sampled.squeeze(2).permute(0, 2, 1)

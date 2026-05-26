import torch
import torch.nn.functional as F


class ObservationOperator:
    def forward(self, x):
        raise NotImplementedError


class InpaintingOperator(ObservationOperator):
    def __init__(self, mask):
        self.mask = mask

    def forward(self, x):
        return x * self.mask


class SuperResolutionOperator(ObservationOperator):
    def __init__(self, scale=4):
        self.scale = scale

    def forward(self, x):
        b, c, h, w = x.shape
        low = F.interpolate(x, size=(h // self.scale, w // self.scale), mode='bicubic', align_corners=False)
        return low


class DenoiseOperator(ObservationOperator):
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def forward(self, x):
        return x + self.sigma * torch.randn_like(x)


class SparseCoordinateOperator(ObservationOperator):
    def __init__(self, coords):
        self.coords = coords

    def forward(self, x):
        grid = self.coords * 2 - 1
        return F.grid_sample(x, grid[:, None], align_corners=True).squeeze(2)

import torch
import torch.nn.functional as F


def cross_resolution_consistency(img_hi, img_lo):
    lo_up = F.interpolate(img_lo, size=img_hi.shape[-2:], mode='bilinear', align_corners=False)
    return F.mse_loss(img_hi, lo_up).item()


def overlap_consistency(values_a, values_b):
    return F.mse_loss(values_a, values_b).item()

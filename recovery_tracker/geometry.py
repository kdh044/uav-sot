"""Original-image coordinates and native convolution feature centers.

Pixel centers are .5+i; k3/s2/p1 convolutions retain origin .5, not stride/2.
Uniform resized ViT maps deliberately keep the ordinary grid_sample transform.
"""
import torch
from torch.nn import functional as F


def feature_grid(points, feature_shape, image_shape=None, stride=None):
    if stride is None:
        return points * 2 - 1
    h, w = image_shape
    fh, fw = feature_shape
    pixels = points * points.new_tensor([w, h])
    index = (pixels - .5) / stride
    return (index + .5) / points.new_tensor([fw, fh]) * 2 - 1


def feature_centers(indices, shape, image_shape=None, stride=None):
    h, w = shape
    xy = torch.stack([indices % w, indices // w], -1)
    if stride is None:
        return (xy + .5) / xy.new_tensor([w, h])
    ih, iw = image_shape
    return (xy * stride + .5) / xy.new_tensor([iw, ih])


def sample_roi(feature, boxes, image_shape=None, stride=None):
    """Shared train/runtime memory encoder. Accept B,4 or B,Q,4 boxes."""
    single = boxes.ndim == 2
    if single:
        boxes = boxes[:, None]
    axis = torch.linspace(-.35, .35, 3, device=feature.device, dtype=feature.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    offsets = torch.stack([xx, yy], -1).reshape(1, 1, 9, 2)
    points = boxes[:, :, None, :2] + offsets * boxes[:, :, None, 2:]
    grid = feature_grid(points, feature.shape[-2:], image_shape, stride)
    pooled = F.grid_sample(feature, grid, align_corners=False).mean(-1).transpose(1, 2)
    return pooled[:, 0] if single else pooled


def resample_uniform(field, shape, image_shape, stride):
    """Sample a uniform image field at actual native-map feature centers."""
    fh, fw = shape
    indices = torch.arange(fh * fw, device=field.device).reshape(1, -1)
    centers = feature_centers(indices, shape, image_shape, stride).to(field.dtype)
    grid = (centers.reshape(1, fh, fw, 2) * 2 - 1).expand(field.shape[0], -1, -1, -1)
    return F.grid_sample(field, grid, align_corners=False, padding_mode='border')

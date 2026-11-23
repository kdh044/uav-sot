"""Observation decoder prototype; validation status is recorded in research reports.

Inputs are aligned detail/context feature maps, identity memory and externally
provided motion hypotheses. Image encoders, motion estimation, training labels,
memory updates and video inference are deliberately outside this module.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import feature_grid


def masked_softmax(logits, valid, dim=-1):
    """All-invalid rows return zeros rather than attending to padded features."""
    scores = logits.masked_fill(~valid, -torch.finfo(logits.dtype).max)
    weights = scores.softmax(dim) * valid.to(logits.dtype)
    return weights / weights.sum(dim, keepdim=True).clamp_min(1e-8)


class ObservationLayer(nn.Module):
    def __init__(self, dim, levels, points):
        super().__init__()
        self.levels, self.points = levels, points
        self.memory_attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm_memory = nn.LayerNorm(dim)
        self.norm_visual = nn.LayerNorm(dim)
        # Geometry: sigma_xy, target_size_xy, identity ambiguity (5 scalars).
        self.condition = nn.Sequential(nn.Linear(dim + 5, dim), nn.GELU())
        self.offset = nn.Linear(dim, levels * points * 2)
        self.point_bias = nn.Linear(dim, levels * points)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.box = nn.Linear(dim, 4)
        self.identity = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1))
        self.no_match = nn.Linear(dim, 1)
        self.write_quality = nn.Linear(dim, 1)

    def forward(self, queries, features, feature_masks, references, sizes,
                sigma, memory, memory_valid, ambiguity, image_shape=None, strides=None):
        b, q, d = queries.shape
        # A first-frame anchor is always supplied, so memory rows cannot be empty.
        evidence, _ = self.memory_attention(queries, memory, memory,
                                            key_padding_mask=~memory_valid, need_weights=False)
        queries = self.norm_memory(queries + evidence)
        geometry = torch.cat([sigma, sizes, ambiguity.expand(b, q, 1)], -1)
        conditioned = self.condition(torch.cat([queries, geometry], -1))
        offsets = self.offset(conditioned).view(b, q, self.levels, self.points, 2).tanh()
        # Detail samples stay at object scale. Context samples cover motion spread.
        radii = sizes[:, :, None, :].expand(-1, -1, self.levels, -1) * .5
        level_context = torch.zeros_like(radii)
        level_context[:, :, -1] = sigma
        radii = radii + level_context
        locations = references[:, :, None, None] + offsets * radii[:, :, :, None]
        sampled, validity = [], []
        for level, (feature, mask) in enumerate(zip(features, feature_masks)):
            grid = feature_grid(locations[:, :, level], feature.shape[-2:],
                                image_shape, None if strides is None else strides[level])
            values = F.grid_sample(feature, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            sampled.append(values.permute(0, 2, 3, 1))
            m = F.grid_sample(mask[:, None].float(), grid, mode='bilinear',
                              padding_mode='zeros', align_corners=False)[:, 0]
            inside = ((locations[:, :, level] >= 0) & (locations[:, :, level] <= 1)).all(-1)
            validity.append((m > .999) & inside)
        values = torch.stack(sampled, 2).reshape(b, q, -1, d)
        valid = torch.stack(validity, 2).reshape(b, q, -1)
        logits = (self.key(values) * queries[:, :, None]).sum(-1) / math.sqrt(d)
        logits = logits + self.point_bias(conditioned)
        weights = masked_softmax(logits, valid)
        observed = (self.value(values) * weights[..., None]).sum(2)
        queries = self.norm_visual(queries + self.output(observed))
        queries = queries + self.ffn(self.ffn_norm(queries))
        delta = self.box(queries)
        centers = references + delta[..., :2].tanh() * (sizes * .5 + sigma)
        box_sizes = sizes * delta[..., 2:].tanh().exp()
        anchor = memory[:, :1].expand(-1, q, -1)
        identity = self.identity(torch.cat([queries, anchor], -1)).squeeze(-1)
        visible = valid.any(-1)
        identity = identity.masked_fill(~visible, -1e4)
        null = self.no_match(queries.mean(1))
        association = torch.cat([identity, null], -1)
        probabilities = association.softmax(-1)
        # Entropy describes ambiguity, not calibrated probability of correctness.
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)
        entropy = (entropy / math.log(q + 1))[:, None, None]
        return queries, centers, box_sizes, entropy, dict(
            boxes_cxcywh=torch.cat([centers, box_sizes], -1),
            association_logits=association, write_logits=self.write_quality(queries).squeeze(-1),
            sample_locations=locations, sample_weights=weights.view(b, q, self.levels, self.points),
            sample_valid=valid.view(b, q, self.levels, self.points), ambiguity=entropy)


class IdentityObservationDecoder(nn.Module):
    """Feature-level prototype with fixed sample budget and recurrent layer evidence.

    Level ordering: high-resolution detail first, low-resolution context last.
    All coordinates and sigmas use the same normalized original-image frame.
    Memory slot zero is the immutable first-target anchor; recent slots may be masked.
    Invalid observations force no-match; recent memory is never mutated here.
    """
    def __init__(self, dim=128, queries=8, levels=3, points=8, layers=3):
        super().__init__()
        self.dim, self.query_count, self.level_count = dim, queries, levels
        self.coupled_observation = True
        self.query_slots = nn.Parameter(torch.randn(1, queries, dim) * .02)
        self.layers = nn.ModuleList([ObservationLayer(dim, levels, points) for _ in range(layers)])

    def forward(self, features, references, sizes, motion_sigma, memory,
                memory_valid=None, feature_masks=None, image_shape=None, strides=None):
        b = memory.shape[0]
        if len(features) != self.level_count or any(x.shape[:2] != (b, self.dim) for x in features):
            raise ValueError('Expected aligned B,D,H,W maps with the configured number of levels')
        for name, value in [('references', references), ('sizes', sizes), ('motion_sigma', motion_sigma)]:
            if value.shape != (b, self.query_count, 2) or not torch.isfinite(value).all():
                raise ValueError(f'{name}: expected finite B,Q,2')
        if (sizes <= 0).any() or (motion_sigma < 0).any():
            raise ValueError('Object sizes must be positive; sigmas must be nonnegative')
        if memory_valid is None:
            memory_valid = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
        if not memory_valid[:, 0].all():
            raise ValueError('First-target memory anchor must be valid')
        if feature_masks is None:
            feature_masks = [torch.ones((b, x.shape[2], x.shape[3]), dtype=torch.bool,
                                       device=x.device) for x in features]
        if len(feature_masks) != self.level_count:
            raise ValueError('One feature validity mask is required per level')
        queries = self.query_slots.expand(b, -1, -1) + memory[:, :1]
        ambiguity = torch.ones((b, 1, 1), device=memory.device, dtype=memory.dtype)
        outputs = []
        for layer in self.layers:
            if not self.coupled_observation:
                ambiguity = torch.zeros_like(ambiguity)
            layer_sigma = motion_sigma if self.coupled_observation else torch.zeros_like(motion_sigma)
            queries, references, sizes, ambiguity, result = layer(
                queries, features, feature_masks, references, sizes, layer_sigma,
                memory, memory_valid, ambiguity, image_shape, strides)
            outputs.append(result)
        return dict(final=outputs[-1], auxiliary=outputs[:-1], query_features=queries)

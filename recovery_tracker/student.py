"""Image-to-query student with a native-resolution stem and frozen OSTrack context.

SAM3 is never imported here. Top-k proposal extraction is discrete; the proposal
head has its own supervision. This is not claimed to be fully differentiable.
"""
from pathlib import Path
import copy
import sys
import torch
from torch import nn
from torch.nn import functional as F
from .query_decoder import IdentityObservationDecoder
from .geometry import sample_roi, feature_centers

SOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTEXT = SOT / 'weights/ostrack/models/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar'


def block(a, b, stride):
    return nn.Sequential(nn.Conv2d(a, b, 3, stride, 1, bias=False),
                         nn.GroupNorm(8, b), nn.GELU())


class DetailEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.early = nn.Sequential(block(3, 24, 2), block(24, 48, 2))
        self.late = block(48, 64, 2)
        self.project4, self.project8 = nn.Conv2d(48, dim, 1), nn.Conv2d(64, dim, 1)

    def forward(self, image):
        a = self.early(image)
        return self.project4(a), self.project8(self.late(a))


class OSTrackContext(nn.Module):
    def __init__(self, checkpoint=DEFAULT_CONTEXT):
        super().__init__()
        source = SOT / 'research_bases/OSTrack'
        if not source.is_dir():
            raise FileNotFoundError(f'Missing pristine official source: {source}')
        if 'lib' in sys.modules:
            origin = Path(sys.modules['lib'].__file__).resolve()
            if source.resolve() not in origin.parents:
                raise RuntimeError('Another tracker has already imported lib; use a separate process')
        sys.path.insert(0, str(source))
        from lib.config.ostrack.config import cfg, update_config_from_file
        from lib.models.ostrack import build_ostrack
        update_config_from_file(str(source / 'experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml'))
        network = build_ostrack(copy.deepcopy(cfg), training=False)
        # Local official checkpoint is trusted; no checkpoints from arbitrary URLs.
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        network.load_state_dict(state['net'], strict=True)
        self.backbone = network.backbone.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, template, image):
        search = F.interpolate(image, (256, 256), mode='bilinear', align_corners=False)
        # Preserve all context tokens. CE-on is a separate future ablation.
        tokens, _ = self.backbone(z=template, x=search, ce_keep_rate=1.0)
        return tokens[:, -256:].transpose(1, 2).reshape(image.shape[0], 768, 16, 16)


def sample_box_feature(feature, boxes):
    """Pool a 3x3 grid inside normalized cxcywh boxes, without clipping to GT."""
    b = feature.shape[0]
    axis = torch.linspace(-.35, .35, 3, device=feature.device, dtype=feature.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    grid = boxes[:, None, None, :2] + torch.stack([xx, yy], -1)[None] * boxes[:, None, None, 2:]
    return F.grid_sample(feature, grid * 2 - 1, align_corners=False).flatten(2).mean(-1)


class CorrelationMotion(nn.Module):
    """Trainable correlation-based apparent-motion prior, not a calibrated camera pose."""
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Conv2d(dim, 16, 1)
        self.log_temperature = nn.Parameter(torch.tensor(2.0))
        self.refine = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 4))

    def forward(self, previous, current):
        n = 24
        a = F.normalize(F.adaptive_avg_pool2d(self.projection(previous), (n, n)).float(), dim=1)
        b = F.normalize(F.adaptive_avg_pool2d(self.projection(current), (n, n)).float(), dim=1)
        correlation = torch.fft.irfft2(torch.fft.rfft2(b) * torch.fft.rfft2(a).conj(), s=(n, n)).sum(1)
        correlation = torch.fft.fftshift(correlation, dim=(-2, -1)) / (n * n)
        logits = correlation.flatten(1) * self.log_temperature.exp().clamp(max=100)
        probabilities = logits.softmax(-1)
        axis = (torch.arange(n, device=a.device).float() - n // 2) / n
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        coordinates = torch.stack([xx.flatten(), yy.flatten()], -1)
        mean = probabilities @ coordinates
        variance = (probabilities[:, :, None] * (coordinates[None] - mean[:, None]).square()).sum(1)
        summary = torch.cat([mean, variance.sqrt()], -1)
        residual = self.refine(summary)
        displacement = mean + .1 * residual[:, :2].tanh()
        sigma = (variance.sqrt() * torch.exp(.5 * residual[:, 2:].tanh())).clamp(.005, .35)
        return displacement, sigma


class RecoveryStudent(nn.Module):
    def __init__(self, dim=64, queries=8, layers=3, points=8, load_context=True,
                 context_checkpoint=DEFAULT_CONTEXT, coupled_observation=True,
                 knowledge_distillation=False,native_proposals=False, repaired_observation=False):
        super().__init__()
        if queries < 4 or queries % 2:
            raise ValueError('Use an even query count >=4')
        self.config = dict(dim=dim, queries=queries, layers=layers, points=points,
                           coupled_observation=coupled_observation)
        if knowledge_distillation:
            self.config['knowledge_distillation'] = True
        if native_proposals:
            self.config['native_proposals'] = True
        if repaired_observation:
            self.config['repaired_observation'] = True
        self.detail = DetailEncoder(dim)
        self.context = OSTrackContext(context_checkpoint) if load_context else None
        self.context_projection = nn.Conv2d(768, dim, 1)
        self.proposal = nn.Sequential(block(dim, 32, 1), nn.Conv2d(32, 1, 1))
        self.foreground = nn.Sequential(block(dim, 32, 1), nn.Conv2d(32, 1, 1))
        self.motion = CorrelationMotion(dim)
        self.decoder = IdentityObservationDecoder(dim, queries, 3, points, layers)
        self.decoder.coupled_observation = coupled_observation
        self.proposal_geometry = nn.Conv2d(dim, 4, 1) if repaired_observation else None
        if repaired_observation:
            nn.init.zeros_(self.proposal_geometry.weight)
            nn.init.zeros_(self.proposal_geometry.bias)
            for layer in self.decoder.layers:
                nn.init.zeros_(layer.box.weight)
                nn.init.zeros_(layer.box.bias)
        self.local_offsets = nn.Parameter(torch.randn(queries // 2, 2) * .2)
        self.identity_transfer = nn.Sequential(nn.Linear(dim,128),nn.GELU(),nn.Linear(128,256)) if knowledge_distillation else None
        self.memory_transfer = nn.Sequential(nn.Linear(dim,128),nn.GELU(),nn.Linear(128,64)) if knowledge_distillation else None

    def encode_memory(self, feature, boxes, image_shape):
        repaired = self.config.get('repaired_observation', False)
        return sample_roi(feature, boxes, image_shape, 4 if repaired else None)

    def encode_context(self, template, current):
        if self.context is None:
            raise RuntimeError('A real frozen context encoder or cached context_features is required')
        return self.context(template, current)

    def forward(self, current, previous, template_detail, template_context, prior_box,
                memory=None, context_features=None):
        fine, mid = self.detail(current)
        previous_fine, previous_mid = self.detail(previous)
        repaired = self.config.get('repaired_observation', False)
        anchor_map, _ = self.detail(template_detail)
        anchor = anchor_map.mean((2, 3))
        if memory is None:
            # Training receives the same imperfect previous-box feature as a runtime can.
            recent = self.encode_memory(previous_fine, prior_box, previous.shape[-2:]) if repaired else sample_box_feature(previous_mid,prior_box)
            memory = torch.stack([anchor,recent],1)
        if context_features is None:
            context_features = self.encode_context(template_context, current)
        context = self.context_projection(context_features)
        displacement, sigma = self.motion(previous_mid, mid)
        conditional_fine=fine*anchor[:,:,None,None] if self.config.get('native_proposals') else fine
        foreground_logits=self.foreground(conditional_fine)
        proposal_logits = self.proposal(context)
        k = self.config['queries'] // 2
        pooled = F.max_pool2d(proposal_logits, 3, 1, 1)
        peaks = proposal_logits.masked_fill(proposal_logits < pooled, -1e4)
        nk=k//2 if self.config.get('native_proposals') else 0
        indices = peaks.flatten(1).topk(k-nk, dim=1).indices
        gh, gw = proposal_logits.shape[-2:]
        global_centers = torch.stack([(indices % gw + .5) / gw, (indices // gw + .5) / gh], -1)
        if nk:
            fine_peaks=foreground_logits.masked_fill(foreground_logits<F.max_pool2d(foreground_logits,3,1,1),-1e4)
            fi=fine_peaks.flatten(1).topk(nk,dim=1).indices
            fh,fw=foreground_logits.shape[-2:]
            fine_centers=feature_centers(fi, (fh,fw), current.shape[-2:], 4 if repaired else None)
            if repaired:
                native_geometry=self.proposal_geometry(conditional_fine)
                selected_geometry=native_geometry.flatten(2).transpose(1,2).gather(1,fi[:,:,None].expand(-1,-1,4))
                fine_centers=fine_centers+selected_geometry[:,:,:2].tanh()*fine_centers.new_tensor([4/current.shape[-1],4/current.shape[-2]])
            global_centers=torch.cat([global_centers,fine_centers],1)
        local = prior_box[:, None, :2] + displacement[:, None] + self.local_offsets[None].tanh() * sigma[:, None]
        references = torch.cat([local, global_centers], 1)
        h, w = current.shape[-2:]
        floor = prior_box.new_tensor([1 / w, 1 / h])
        sizes = torch.maximum(prior_box[:, None, 2:], floor).expand(-1, k * 2, -1)
        spread = sigma[:, None].expand(-1, k * 2, -1)
        if repaired and nk:
            sizes=torch.cat([sizes[:,:-nk], sizes[:,-nk:]*selected_geometry[:,:,2:].tanh().exp()],1)
        decoded = self.decoder([fine, mid, context], references, sizes, spread, memory,
                               image_shape=current.shape[-2:], strides=[4,8,None] if repaired else None)
        if repaired:
            decoded.update(image_shape=current.shape[-2:], native_stride=4,
                           proposal_geometry=self.proposal_geometry(conditional_fine),
                           input_memory=memory)
        decoded.update(foreground_logits=foreground_logits, proposal_logits=proposal_logits,
                       displacement=displacement, motion_sigma=sigma,
                       anchor_feature=anchor, detail_feature=fine)
        if self.identity_transfer is not None and self.training:
            occupancy=foreground_logits.sigmoid()
            pool=(fine*occupancy).sum((2,3))/occupancy.sum((2,3)).clamp_min(1e-8)
            if repaired:
                pool=self.encode_memory(fine,decoded['final']['boxes_cxcywh'],current.shape[-2:])
            decoded.update(distilled_queries=self.identity_transfer(decoded['query_features']),
                           distilled_anchor=self.identity_transfer(anchor),
                           distilled_memory=self.memory_transfer(pool),memory_pool_feature=pool)
        return decoded

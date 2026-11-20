"""실험(2026-11-12): 주변시(peripheral) + 중심시(foveal) ORTrack.

one-stream ViT 입력 = [template 64 | peripheral G*G | placement 1 | search 256] (search 를 마지막에 두어 원본 box head 그대로 사용).
- peripheral: 전체 프레임을 RES x RES 로 축소 → 원본 patch_embed(16) → POOL x POOL 평균 → G x G 토큰
- 좌표 부호화: 토큰마다 전체 프레임 정규화 좌표(u,v)와 현재 crop 기준 상대좌표를 MLP 로 embed (search 토큰에도 같은 MLP, 마지막 층 0 초기화)
- heads: (a) 원본 box head (search) / (b) placement 토큰 → 다음 프레임 대상 중심 (현재 crop 중심 기준, crop 한 변 단위) + log 크기
         (c) placement 토큰 → 존재(대상 중심이 현재 crop 안) logit / (d) 주변시 heatmap(G*G) + 셀 내부 offset → 현재 프레임 대상 위치
모든 shape 고정 (데이터 의존 분기 없음).
pcoord: (B,4) = [crop 중심 u, v (전체 프레임 정규화), crop 한 변/W, crop 한 변/H]
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

from lib.models.ortrack.ortrack import ORTrack


def _fourier(u, nf=4):
    # u: (..., 2) in [0,1] → (..., 4*nf)
    k = (2.0 ** torch.arange(nf, device=u.device, dtype=u.dtype)) * math.pi
    a = u.unsqueeze(-1) * k                                  # (...,2,nf)
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


class PeriphORTrack(ORTrack):
    def __init__(self, transformer, box_head, head_type, pcfg):
        super().__init__(transformer, box_head, aux_loss=False, head_type=head_type)
        D = transformer.embed_dim
        self.p_res, self.p_pool = int(pcfg.RES), int(pcfg.POOL)
        self.p_grid = G = self.p_res // 16 // self.p_pool
        self.n_p = G * G
        self.p_merge = getattr(pcfg, 'MERGE', 'avg')
        if self.p_merge == 'linear' and self.p_pool > 1:
            k = self.p_pool ** 2
            self.periph_merge = nn.Linear(k * D, D)
            with torch.no_grad():  # 평균으로 초기화
                self.periph_merge.weight.copy_(torch.cat([torch.eye(D) / k] * k, 1)); self.periph_merge.bias.zero_()
        self.periph_pos = nn.Parameter(torch.zeros(1, self.n_p, D))
        self.periph_type = nn.Parameter(torch.zeros(1, 1, D))
        self.periph_place_token = nn.Parameter(torch.randn(1, 1, D) * .02)
        self.n_feat = 4 * 4 + 2 + 2 + 1
        self.periph_coord = nn.Sequential(nn.Linear(self.n_feat, 128), nn.GELU(), nn.Linear(128, D))
        nn.init.zeros_(self.periph_coord[2].weight); nn.init.zeros_(self.periph_coord[2].bias)
        self.periph_cls = nn.Linear(D, 1)
        self.periph_off = nn.Linear(D, 2)
        self.periph_place = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 4))  # du, dv, ls_next, presence
        # 고정 격자 (buffer)
        g = (torch.arange(G, dtype=torch.float32) + .5) / G
        gy, gx = torch.meshgrid(g, g, indexing='ij')
        self.register_buffer('grid_p', torch.stack([gx, gy], -1).view(1, -1, 2), persistent=False)   # [0,1]
        S = self.feat_sz_s
        s = (torch.arange(S, dtype=torch.float32) + .5) / S - .5
        sy, sx = torch.meshgrid(s, s, indexing='ij')
        self.register_buffer('grid_x', torch.stack([sx, sy], -1).view(1, -1, 2), persistent=False)   # [-.5,.5]

    def init_periph_pos(self):
        """원본 search pos_embed(16x16)를 G x G 로 보간해 초기화 (pretrained 로드 후 호출)."""
        pe = self.backbone.pos_embed_x.data
        S = int(round(pe.shape[1] ** .5))
        pe = pe.transpose(1, 2).reshape(1, -1, S, S)
        pe = F.interpolate(pe, size=(self.p_grid, self.p_grid), mode='bilinear', align_corners=False)
        self.periph_pos.data.copy_(pe.flatten(2).transpose(1, 2))

    def _coord_feat(self, u, rel, c):
        # u: (B,N,2) 전체 프레임 정규화 좌표, rel: (B,N,2) crop 한 변 단위 상대좌표, c: (B,4)
        B, N = u.shape[:2]
        inside = ((rel.abs() <= .5).all(-1, keepdim=True)).to(u.dtype)
        logs = torch.log(c[:, None, 2:4].clamp(min=1e-4)).expand(B, N, 2) / 3.0
        return torch.cat([_fourier(u.clamp(-1, 2)), rel.clamp(-8, 8) / 8.0, logs, inside], -1)

    def _tokens(self, z, x, p, c):
        bb = self.backbone
        B = x.shape[0]
        x = bb.patch_embed(x) + bb.pos_embed_x
        z = bb.patch_embed(z) + bb.pos_embed_z
        p = bb.patch_embed(p)                                       # (B, (RES/16)^2, D)
        D = p.shape[-1]; R = self.p_res // 16
        if self.p_pool > 1 and self.p_merge == 'linear':
            k, G = self.p_pool, R // self.p_pool       # space-to-depth: (B,G,k,G,k,D) → (B,G*G,k*k*D)
            p = p.view(B, G, k, G, k, D).permute(0, 1, 3, 2, 4, 5).reshape(B, G * G, k * k * D)
            p = self.periph_merge(p)
        elif self.p_pool > 1:
            p = F.avg_pool2d(p.transpose(1, 2).reshape(B, D, R, R), self.p_pool).flatten(2).transpose(1, 2)
        up = self.grid_p.expand(B, -1, -1)
        relp = (up - c[:, None, :2]) / c[:, None, 2:4]
        p = p + self.periph_pos + self.periph_type + self.periph_coord(self._coord_feat(up, relp, c))
        relx = self.grid_x.expand(B, -1, -1)
        ux = c[:, None, :2] + relx * c[:, None, 2:4]
        x = x + self.periph_coord(self._coord_feat(ux, relx, c))
        q = self.periph_place_token.expand(B, -1, -1)
        t = torch.cat([z, p, q, x], 1)
        t = bb.pos_drop(t)
        for blk in bb.blocks:
            t = blk(t)
        return bb.norm(t)

    def forward(self, template, search, periph=None, pcoord=None, is_distill=False):
        if periph is None:
            return super().forward(template, search, is_distill=is_distill)
        t = self._tokens(template, search, periph, pcoord)
        if self.training:
            mask = self.masking(template, 16, 0.3, template.device) if self.randomMask else None
            if mask is None:
                import numpy as np
                from scipy.stats import multivariate_normal
                if len(self.intensity) == 0:
                    r = int(template.shape[-1] / 2); sigma = 64
                    xg, yg = np.mgrid[-r:r:1, -r:r:1]
                    inten = multivariate_normal([0.0, 0.], [[sigma * r, 0.0], [0.0, sigma * r]]).pdf(np.dstack((xg, yg)))
                    self.intensity = inten / inten.sum()
                mask = self.masking_CoxProcess(template.shape[0], self.intensity, 16, 0.3, template.device)
            mask = mask.repeat(1, template.shape[1], 1, 1)
            t1 = self._tokens(template * mask, search, periph, pcoord)
            sim_loss = F.mse_loss(t[:, :self.feat_len_t], t1[:, :self.feat_len_t].detach())
        else:
            sim_loss = 0
        out = self.forward_head(t, None)
        n0 = self.feat_len_t
        pt = t[:, n0:n0 + self.n_p]
        qt = t[:, n0 + self.n_p]
        out['periph_logits'] = self.periph_cls(pt).squeeze(-1)          # (B, G*G)
        out['periph_off'] = torch.sigmoid(self.periph_off(pt))           # (B, G*G, 2)
        pl = self.periph_place(qt)
        out['place_dxy'] = pl[:, :2]                                     # 다음 중심 - crop 중심, crop 한 변 단위
        out['place_ls'] = pl[:, 2]                                       # log(sqrt(next wh) / crop 기준 크기)
        out['presence_logit'] = pl[:, 3]
        out['sim_loss'] = sim_loss
        out['backbone_feat'] = t
        return out

    def periph_estimate(self, out):
        """주변시 현재 위치 추정 (전체 프레임 정규화 u,v) 과 최대 확률."""
        G = self.p_grid
        prob = out['periph_logits'].softmax(-1)
        conf, idx = prob.max(-1)
        off = out['periph_off'][torch.arange(idx.shape[0], device=idx.device), idx]
        u = ((idx % G).to(off.dtype) + off[:, 0]) / G
        v = ((idx // G).to(off.dtype) + off[:, 1]) / G
        return torch.stack([u, v], -1), conf


class SidePeriphORTrack(PeriphORTrack):
    """실험(2026-11-13): '손해 없는' 주변시 — 원본 ORTrack(템플릿+search one-stream)은 그대로 동결하고,
    주변시는 별도 가지 [template 64 | peripheral G*G | placement 1] 로 처리한다.
    - 공유: 동결된 patch_embed, pos_embed_z, 앞쪽 12-K block
    - 학습: 마지막 K block 복사본(periph_blocks), periph_* 모듈, 존재 head(periph_pres: placement 토큰 + 국소 search 특징 평균 + score 최대)
    재획득을 하지 않는 프레임의 박스 출력은 원본 ORTrack 과 비트 단위로 같다(동결 + eval 모드)."""
    def __init__(self, transformer, box_head, head_type, pcfg):
        super().__init__(transformer, box_head, head_type, pcfg)
        import copy
        K = int(getattr(pcfg, 'SIDE_BLOCKS', 2))
        self.side_k = K
        self.periph_blocks = nn.ModuleList([copy.deepcopy(b) for b in transformer.blocks[-K:]])
        self.periph_norm = copy.deepcopy(transformer.norm)
        D = transformer.embed_dim
        self.periph_pres = nn.Sequential(nn.Linear(2 * D + 1, D), nn.GELU(), nn.Linear(D, 1))

    def sync_side_from_backbone(self):
        """pretrained 로드 후 호출: 복사 block 을 원본 마지막 K block 가중치로 초기화."""
        K = self.side_k
        for d, s in zip(self.periph_blocks, self.backbone.blocks[-K:]):
            d.load_state_dict(s.state_dict())
        self.periph_norm.load_state_dict(self.backbone.norm.state_dict())

    def freeze_local(self):
        for n, p in self.named_parameters():
            p.requires_grad = n.startswith('periph_')

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval(); self.box_head.eval()   # 동결 부분은 항상 eval (drop path 등 끔)
        return self

    def _side_tokens(self, z, p, c):
        bb = self.backbone
        B = z.shape[0]
        z = bb.patch_embed(z) + bb.pos_embed_z
        p = bb.patch_embed(p)
        D = p.shape[-1]; R = self.p_res // 16
        if self.p_pool > 1:
            p = F.avg_pool2d(p.transpose(1, 2).reshape(B, D, R, R), self.p_pool).flatten(2).transpose(1, 2)
        up = self.grid_p.expand(B, -1, -1)
        relp = (up - c[:, None, :2]) / c[:, None, 2:4]
        p = p + self.periph_pos + self.periph_type + self.periph_coord(self._coord_feat(up, relp, c))
        q = self.periph_place_token.expand(B, -1, -1)
        t = torch.cat([z, p, q], 1)
        for blk in bb.blocks[:len(bb.blocks) - self.side_k]:
            t = blk(t)
        for blk in self.periph_blocks:
            t = blk(t)
        return self.periph_norm(t)

    def forward(self, template, search, periph=None, pcoord=None, is_distill=False):
        with torch.no_grad():
            x, _ = self.backbone(z=template, x=search)
            out = self.forward_head(x, None)
        if periph is None:
            out['sim_loss'] = torch.zeros((), device=search.device)
            return out
        t = self._side_tokens(template, periph, pcoord)
        n0 = self.feat_len_t
        pt = t[:, n0:n0 + self.n_p]; qt = t[:, n0 + self.n_p]
        out['periph_logits'] = self.periph_cls(pt).squeeze(-1)
        out['periph_off'] = torch.sigmoid(self.periph_off(pt))
        pl = self.periph_place(qt)
        out['place_dxy'] = pl[:, :2]; out['place_ls'] = pl[:, 2]
        loc = x[:, n0:].mean(1).detach()
        smax = out['score_map'].flatten(1).max(-1, keepdim=True).values.detach()
        out['presence_logit'] = self.periph_pres(torch.cat([qt, loc, smax], -1)).squeeze(-1)
        out['sim_loss'] = torch.zeros((), device=search.device)
        out['backbone_feat'] = x
        return out

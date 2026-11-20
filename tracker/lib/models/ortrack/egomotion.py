"""실험(2026-11-12): 주변시 ego-motion 네트워크 (CMC 대체, 자기지도).

입력: 이전/현재 전체 프레임 회색 저해상도 (B,1,R,R) 두 장 (R=256, 주변시와 같은 비등방 축소).
구조: 공유 CNN encoder (stride 8 → 32x32xC) → 전역 상관(행렬곱, softmax, 학습 온도) → 이전 프레임 각 셀의 현재 위치 기대값(flow)
      + 셀별 신뢰 가중치(conv head) → 가중 최소제곱으로 호모그래피(8 dof, DLT). 네트워크 그래프는 고정 shape
      (conv / matmul / softmax); 마지막 8x8 선형계 풀이는 호스트(또는 torch.linalg.solve)에서 1회.
좌표: 정규화 [0,1]^2, H 는 이전→현재.
"""
import torch
from torch import nn
import torch.nn.functional as F


def _cbr(i, o, s):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class EgoMotionNet(nn.Module):
    def __init__(self, res=256, ch=96):
        super().__init__()
        self.res = res
        self.enc = nn.Sequential(_cbr(1, 32, 2), _cbr(32, 32, 1), _cbr(32, 64, 2), _cbr(64, 64, 1),
                                 _cbr(64, ch, 2), _cbr(ch, ch, 1), nn.Conv2d(ch, ch, 3, 1, 1))
        self.g = g = res // 8
        self.log_temp = nn.Parameter(torch.tensor(2.5))
        self.wnet = nn.Sequential(nn.Conv2d(ch + 4, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, 1, 1),
                                  nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1))
        c = (torch.arange(g, dtype=torch.float32) + .5) / g
        yy, xx = torch.meshgrid(c, c, indexing='ij')
        self.register_buffer('grid', torch.stack([xx, yy], -1).view(-1, 2), persistent=False)   # (g*g,2)

    def forward(self, a, b):
        """a,b: (B,1,R,R) [0,1]. 반환 flow 목적지 (B,N,2), 가중치 (B,N), 상관 최대확률 (B,N)."""
        B = a.shape[0]
        f = self.enc(torch.cat([a, b], 0) - 0.45)
        fa, fb = f[:B], f[B:]
        C = fa.shape[1]
        fa_, fb_ = F.normalize(fa.flatten(2), dim=1), F.normalize(fb.flatten(2), dim=1)   # (B,C,N)
        corr = torch.bmm(fa_.transpose(1, 2), fb_) * self.log_temp.exp() ** 2                # (B,N,N)
        p = corr.softmax(-1)
        dst = torch.bmm(p, self.grid.expand(B, -1, -1))                                      # (B,N,2)
        pmax = p.max(-1).values
        g = self.g
        flow = (dst - self.grid).transpose(1, 2).reshape(B, 2, g, g)
        extra = torch.cat([flow * 8, pmax.view(B, 1, g, g), (p * (p + 1e-9).log()).sum(-1).view(B, 1, g, g) / 7.0], 1)
        w = self.wnet(torch.cat([fa, extra], 1)).view(B, -1)
        return dst, 0.05 + 0.95 * torch.sigmoid(w), pmax   # 가중치 하한 (붕괴 방지)

    def homography(self, dst, w):
        """가중 DLT (h33=1). 좌표를 [-1,1] 로 옮겨 조건수 개선 후 정규화 [0,1] 좌표의 H 반환 (B,3,3)."""
        B, N, _ = dst.shape
        s = self.grid.expand(B, -1, -1) * 2 - 1
        d = dst * 2 - 1
        x, y, u, v = s[..., 0], s[..., 1], d[..., 0], d[..., 1]
        z, o = torch.zeros_like(x), torch.ones_like(x)
        r1 = torch.stack([x, y, o, z, z, z, -u * x, -u * y], -1)
        r2 = torch.stack([z, z, z, x, y, o, -v * x, -v * y], -1)
        A = torch.cat([r1, r2], 1)                       # (B,2N,8)
        bb = torch.cat([u, v], 1)                        # (B,2N)
        W = torch.cat([w, w], 1) + 1e-6
        AtW = A.transpose(1, 2) * W[:, None]
        # 항등 변환 쪽 ridge (가중치가 모두 0 으로 붕괴할 때 퇴화 방지)
        lam = 1e-4 * W.sum(1).view(B, 1, 1)   # 가중치 합에 비례 (scale 불변)
        hid = torch.tensor([1., 0, 0, 0, 1, 0, 0, 0], device=A.device)
        M = AtW @ A + lam * torch.eye(8, device=A.device)[None]
        h = torch.linalg.solve(M, (AtW @ bb[..., None]) + lam * hid[None, :, None])[..., 0]
        Hc = torch.cat([h, torch.ones(B, 1, device=h.device)], 1).view(B, 3, 3)
        T = torch.tensor([[2., 0, -1], [0, 2, -1], [0, 0, 1]], device=h.device)   # [0,1] → [-1,1]
        Hn = torch.linalg.inv(T) @ Hc @ T
        return Hn / Hn[:, 2:3, 2:3]


def warp_points(H, p):
    """H (B,3,3), p (B,K,2) → (B,K,2)."""
    q = torch.cat([p, torch.ones_like(p[..., :1])], -1) @ H.transpose(1, 2)
    return q[..., :2] / q[..., 2:3].clamp(min=1e-6)

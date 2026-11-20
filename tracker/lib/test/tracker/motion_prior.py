"""검색 영역 중심 사전 예측 (실험용, env SOT_MOTION 으로만 켜짐).

- cmc : 이전→현재 프레임 카메라 움직임을 호모그래피로 추정해 이전 박스 중심을 옮김
        (축소 회색영상, Shi-Tomasi 코너 + LK optical flow + RANSAC, 대상 박스 주변 마스킹)
- cv  : 대상의 등속 운동 (카메라 보정 후 좌표계에서 직전 프레임 변위를 그대로 더함)
- cmc+cv : 둘 다
"""
import os
import cv2
import numpy as np


class MotionPrior:
    def __init__(self, mode, work_width=640):
        self.use_cmc = 'cmc' in mode
        self.use_cv = 'cv' in mode
        self.work_width = work_width
        self.prev_gray = None
        self.scale = 1.0
        self.vel = np.zeros(2)      # 보정 좌표계에서 대상 중심 속도 (px/frame, 원본 해상도)
        self.prev_center = None     # 직전 프레임 최종 추정 중심
        self.cum_angle = 0.0        # 첫 프레임 대비 누적 카메라 회전 (rad)
        self.H = None

    def _gray(self, image):
        h, w = image.shape[:2]
        self.scale = min(1.0, self.work_width / w)
        g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if self.scale < 1.0:
            it = cv2.INTER_NEAREST if os.environ.get('SOT_CMC_INTERP') == 'nearest' else cv2.INTER_AREA  # nearest: 축소 비용 거의 0
            g = cv2.resize(g, (int(w * self.scale), int(h * self.scale)), interpolation=it)
        return g

    def init(self, image, box):
        self.prev_gray = self._gray(image)
        self.prev_center = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])
        self.vel[:] = 0
        self.cum_angle = 0.0

    def _homography(self, gray, box):
        s = self.scale
        mask = np.full(self.prev_gray.shape, 255, np.uint8)
        x, y, w, h = [v * s for v in box]
        cv2.rectangle(mask, (int(x - w), int(y - h)), (int(x + 2 * w), int(y + 2 * h)), 0, -1)
        # 빠른 모드(env SOT_CMC_FAST=1): 직전 프레임에서 추적된 inlier 점을 재사용, 부족할 때만 재검출
        fast = bool(os.environ.get('SOT_CMC_FAST'))
        p0 = None
        if fast and getattr(self, 'pts', None) is not None and len(self.pts) >= 80:
            p0 = self.pts
        if p0 is None:
            p0 = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=int(os.environ.get('SOT_CMC_CORNERS', 200 if fast else 400)), qualityLevel=0.01,
                                         minDistance=8, mask=mask)
        if p0 is None or len(p0) < 12:
            self.pts = None
            return None
        wv = int(os.environ.get('SOT_CMC_WIN', 15 if fast else 21)); win = (wv, wv)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, winSize=win, maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < 12:
            self.pts = None
            return None
        Hm, inl = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 3.0)
        if Hm is None or inl.sum() < 10:
            self.pts = None
            return None
        if fast:  # 다음 프레임용: 대상 주변 제외한 inlier 점
            q = p1[ok][inl.ravel() == 1].reshape(-1, 2)
            keep = ~((q[:, 0] > x - w) & (q[:, 0] < x + 2 * w) & (q[:, 1] > y - h) & (q[:, 1] < y + 2 * h))
            self.pts = q[keep].reshape(-1, 1, 2).astype(np.float32)
        S = np.diag([s, s, 1.0])
        return np.linalg.inv(S) @ Hm @ S  # 원본 해상도 좌표계

    def predict(self, image, box):
        """현재 프레임에서 검색할 박스(x,y,w,h) 반환. box 는 직전 프레임 추정."""
        gray = self._gray(image)
        H_img, W_img = image.shape[:2]
        if not np.all(np.isfinite(box)) or box[2] > W_img or box[3] > H_img:  # 박스 폭주 방지
            box = [box[0], box[1], min(max(box[2], 1), W_img), min(max(box[3], 1), H_img)]
        c = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])
        self.H = None
        if self.use_cmc:
            Hm = self._homography(gray, box)
            if Hm is not None:
                p = Hm @ np.array([c[0], c[1], 1.0])
                c = p[:2] / p[2]
                self.H = Hm
                self.cum_angle += np.arctan2(Hm[1, 0], Hm[0, 0])
        if self.use_cv:
            c = c + self.vel
        # 퇴화 호모그래피로 중심이 화면 밖 멀리 튀면 화면 안으로 제한 (패딩 폭주 방지)
        c = np.array([min(max(c[0], 0.0), W_img - 1.0), min(max(c[1], 0.0), H_img - 1.0)])
        self.prev_gray = gray
        self.cmc_center = c - (self.vel if self.use_cv else 0)  # 카메라 보정만 한 직전 중심
        return [c[0] - box[2] / 2, c[1] - box[3] / 2, box[2], box[3]]

    def update(self, new_box):
        nc = np.array([new_box[0] + new_box[2] / 2, new_box[1] + new_box[3] / 2])
        if self.use_cv:
            v = nc - self.cmc_center
            self.vel = 0.5 * self.vel + 0.5 * v   # 지수평활 속도
        self.prev_center = nc


class LearnedMotion:
    """실험(2026-11-12): 주변시 ego-motion 네트워크(EgoMotionNet)로 CMC 대체 (env SOT_MOTION=learned, SOT_EGO_CKPT).
    전체 프레임 → 256x256 회색 (비등방 축소) 두 장 → 정규화 좌표 호모그래피 → 직전 중심 이동."""
    _net = None

    def __init__(self, ckpt, res=256):
        import torch
        from lib.models.ortrack.egomotion import EgoMotionNet
        if LearnedMotion._net is None:
            net = EgoMotionNet(res=res)
            net.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['net'])
            LearnedMotion._net = net.cuda().eval()
        self.net = LearnedMotion._net
        self.res = res
        self.use_cmc = False; self.use_cv = False
        self.H = None; self.cum_angle = 0.0; self.prev = None

    def _small(self, image):
        import torch
        g = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (self.res, self.res), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(g).cuda().float()[None, None] / 255.

    def init(self, image, box):
        self.prev = self._small(image)

    def predict(self, image, box):
        import torch
        Hh, Ww = image.shape[:2]
        cur = self._small(image)
        with torch.no_grad():
            dst, w, _ = self.net(self.prev, cur)
            Hn = self.net.homography(dst, w)[0].double().cpu().numpy()
        self.prev = cur
        c = np.array([(box[0] + box[2] / 2) / Ww, (box[1] + box[3] / 2) / Hh, 1.0])
        p = Hn @ c
        if np.isfinite(p).all() and abs(p[2]) > 1e-6:
            c = p[:2] / p[2]
        c = np.array([min(max(c[0] * Ww, 0.0), Ww - 1.0), min(max(c[1] * Hh, 0.0), Hh - 1.0)])
        self.H = Hn
        return [c[0] - box[2] / 2, c[1] - box[3] / 2, box[2], box[3]]

    def update(self, new_box):
        pass


class CMCKalman:
    """실험(2026-11-13): CMC(호모그래피)로 카메라 움직임을 뺀 좌표에서 등속 KF. 상태 [cx, cy, vx, vy] (원본 픽셀).
    predict(image, box): 카메라 보정 후 KF 예측 → 검색 박스. update(box): 측정 갱신. d2(box): 예측 대비 Mahalanobis 거리²."""
    def __init__(self, work_width=640, q=1.0, r=0.05):
        self.cmc = MotionPrior('cmc', work_width)
        self.q, self.r = q, r
        self.x = None; self.P = None; self.H = None; self.cam = 0.0

    def init(self, image, box):
        self.cmc.init(image, box)
        s = max((box[2] * box[3]) ** .5, 4.0)
        self.x = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2, 0., 0.]); self.P = np.diag([s, s, s, s]) ** 2 * .1

    def predict(self, image, box):
        pb = self.cmc.predict(image, box); self.H = self.cmc.H
        c0 = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2]); c1 = np.array([pb[0] + pb[2] / 2, pb[1] + pb[3] / 2])
        self.cam = float(np.linalg.norm(c1 - c0))                     # 카메라 움직임에 의한 중심 이동 (px)
        self.x[:2] = c1 if self.x is not None else c1
        if self.H is not None:                                       # 속도도 카메라 회전·축척만큼 변환
            self.x[2:] = self.H[:2, :2] @ self.x[2:]
        F = np.eye(4); F[0, 2] = F[1, 3] = 1
        s = max((box[2] * box[3]) ** .5, 4.0)
        Q = np.diag([.25, .25, 1., 1.]) * (self.q * s * .1) ** 2
        self.x = F @ self.x; self.P = F @ self.P @ F.T + Q
        self.s = s
        return [self.x[0] - box[2] / 2, self.x[1] - box[3] / 2, box[2], box[3]]

    def _S(self):
        R = np.eye(2) * (self.r * self.s) ** 2
        return self.P[:2, :2] + R

    def d2(self, box):
        z = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2]) - self.x[:2]
        return float(z @ np.linalg.solve(self._S(), z))

    def update(self, box):
        z = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])
        Hm = np.zeros((2, 4)); Hm[0, 0] = Hm[1, 1] = 1
        S = self._S(); K = self.P @ Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.x[:2]); self.P = (np.eye(4) - K @ Hm) @ self.P
        self.cmc.update(box)


class PhaseCorrMotion:
    """경량 CMC 대체(2026-11-13): 축소 흑백 프레임(기본 128 폭)의 위상 상관으로 전체 평행 이동만 추정.
    MotionPrior('cmc') 와 같은 인터페이스(init/predict/update, .H). 회전·확대는 무시. 응답(신뢰도)이 낮으면 이동 0."""
    def __init__(self, work_width=128, min_resp=0.05):
        self.w = work_width; self.min_resp = min_resp
        self.prev = None; self.H = None; self.win = None; self.resp = 0.0; self.use_cmc = True

    def _small(self, image):
        h, w = image.shape[:2]; s = self.w / w
        g = cv2.cvtColor(cv2.resize(image, (self.w, max(8, int(round(h * s)))), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        self.scale = s
        return np.float32(g)

    def init(self, image, box):
        self.prev = self._small(image)
        self.win = cv2.createHanningWindow(self.prev.shape[::-1], cv2.CV_32F)

    def predict(self, image, box):
        cur = self._small(image)
        (dx, dy), r = cv2.phaseCorrelate(self.prev, cur, self.win)
        self.prev = cur; self.resp = float(r)
        if r < self.min_resp:
            dx = dy = 0.0
        dx /= self.scale; dy /= self.scale
        self.H = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])
        return [box[0] + dx, box[1] + dy, box[2], box[3]]

    def update(self, new_box):
        pass

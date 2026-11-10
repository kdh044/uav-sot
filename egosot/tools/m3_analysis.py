"""M3: 움직임 예측 위치와 ViT 출력 위치의 불일치가 실패(IoU=0) 신호로 쓸모 있는지.

입력: SOT_LOG_DIR 로 저장된 프레임 로그 (frame, search_cx, search_cy, out_cx, out_cy, score, H_ok, cum_angle, w, h)
      + 결과 박스 폴더 + GT (SOT_BIODRONE 로 분할 지정)
신호: jump = |out - search_center| / sqrt(w*h)   (search_center = CMC+CV 예측 위치)
      score = 점수맵 최댓값
출력: AUROC, 오경보 1%/5% 에서 실패 프레임 탐지율, 놓침 시작 5프레임 안 탐지 비율

usage: SOT_BIODRONE=... python tools/m3_analysis.py <log_dir> <result_view_dir>
"""
import os, sys
import numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ORTrack'))
from lib.test.evaluation import get_dataset
from lib.test.analysis.extract_results import calc_iou_overlap
from lib.test.utils.load_text import load_text

log_dir, res_dir = sys.argv[1:3]
rows = []  # seq_idx, frame, jump, score, lost, onset
for si, seq in enumerate(get_dataset('biodrone')):
    lp = os.path.join(log_dir, seq.name + '.npy')
    if not os.path.exists(lp):
        continue
    L = np.load(lp)
    gt = np.asarray(seq.ground_truth_rect, float)
    pr = load_text(os.path.join(res_dir, seq.name + '.txt'), delimiter=('\t', ','), dtype=np.float64)
    iou = calc_iou_overlap(torch.tensor(pr), torch.tensor(gt)).numpy()
    valid = (gt[:, 2] > 0) & (gt[:, 3] > 0)
    prev_lost = False
    for r in L:
        t = int(r[0])
        if t >= len(gt) or not valid[t]:
            continue
        size = max(np.sqrt(r[8] * r[9]), 1.0)
        jump = np.hypot(r[3] - r[1], r[4] - r[2]) / size
        lost = iou[t] <= 0
        rows.append((si, t, jump, r[5], lost, lost and not prev_lost))
        prev_lost = lost
a = np.array(rows, dtype=float)
lost = a[:, 4] > 0


def auroc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(len(s))
    p = y.sum(); n = len(y) - p
    return (r[y].sum() - p * (p - 1) / 2) / (p * n)


def det_at(s, y, fa):
    th = np.quantile(s[~y], 1 - fa)
    return 100 * (s[y] > th).mean(), th


sigs = {'score (낮을수록 실패)': -a[:, 3], 'jump': a[:, 2],
        'jump × (1-score)': a[:, 2] * (1 - a[:, 3])}
print(f"frames {len(a)}  lost {lost.sum()} ({100*lost.mean():.1f}%)")
for k, s in sigs.items():
    d1, _ = det_at(s, lost, 0.01); d5, th = det_at(s, lost, 0.05)
    on = a[:, 5] > 0
    print(f"{k:22s} AUROC {auroc(s, lost):.3f}  det@1%FA {d1:5.1f}%  det@5%FA {d5:5.1f}%  onset det@5%FA {100*(s[on] > th).mean():5.1f}%")

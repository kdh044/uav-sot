"""놓침 장면 시각화: 놓침 시작 전후 프레임을 한 줄로 이어 붙인 PNG.

초록=GT, 빨강=예측, 노랑=직전 예측 기준 검색영역(한 변 4*sqrt(w*h)), 파랑(선택)=비교 트래커 예측.
각 패널은 GT·예측·검색영역을 모두 포함하도록 잘라서 같은 크기로 확대.

usage: python tools/render_failures.py <out.png> <dataset> <seq> <start> <result_dir> [--cmp <result_dir2>] [--title txt]
"""
import argparse, os, sys
import numpy as np, cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ORTrack'))
from lib.test.evaluation import get_dataset
from lib.test.utils.load_text import load_text

p = argparse.ArgumentParser()
p.add_argument('out'); p.add_argument('dataset'); p.add_argument('seq'); p.add_argument('start', type=int)
p.add_argument('result_dir'); p.add_argument('--cmp', default=None); p.add_argument('--title', default='')
p.add_argument('--offsets', default='-8,-2,0,3,10,30')
p.add_argument('--panel', type=int, default=300)
a = p.parse_args()

seq = [s for s in get_dataset(a.dataset) if s.name == a.seq][0]
gt = np.asarray(seq.ground_truth_rect, float)
pr = load_text(os.path.join(a.result_dir, a.seq + '.txt'), delimiter=('\t', ','), dtype=np.float64)
cmp = load_text(os.path.join(a.cmp, a.seq + '.txt'), delimiter=('\t', ','), dtype=np.float64) if a.cmp else None
n = min(len(gt), len(pr))


def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[0] + b1[2], b2[0] + b2[2]), min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / u if u > 0 else 0


panels = []
for off in map(int, a.offsets.split(',')):
    t = a.start + off
    if t < 1 or t >= n:
        continue
    img = cv2.imread(seq.frames[t])
    H, W = img.shape[:2]
    g, q, prev = gt[t], pr[t], pr[t - 1]
    side = 4 * np.sqrt(max(prev[2] * prev[3], 1))
    pc = prev[:2] + prev[2:] / 2
    sr = [pc[0] - side / 2, pc[1] - side / 2, side, side]
    boxes = [q, sr] + ([g] if g[2] > 0 else []) + ([cmp[t]] if cmp is not None else [])
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2 * 1.15 + 20
    X0, Y0 = int(cx - half), int(cy - half)
    S = int(2 * half)
    canvas = np.zeros((S, S, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(X0, 0), max(Y0, 0), min(X0 + S, W), min(Y0 + S, H)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = img[sy0:sy1, sx0:sx1]
    k = a.panel / S
    canvas = cv2.resize(canvas, (a.panel, a.panel), interpolation=cv2.INTER_NEAREST if k > 3 else cv2.INTER_LINEAR)

    def rect(b, c, th=2):
        cv2.rectangle(canvas, (int((b[0] - X0) * k), int((b[1] - Y0) * k)),
                      (int((b[0] + b[2] - X0) * k), int((b[1] + b[3] - Y0) * k)), c, th)
    rect(sr, (0, 220, 255), 1)
    if g[2] > 0:
        rect(g, (0, 220, 0))
    if cmp is not None:
        rect(cmp[t], (255, 140, 0))
    rect(q, (0, 0, 255))
    lab = f"t{off:+d}  IoU {iou(q, g):.2f}" if g[2] > 0 else f"t{off:+d}  (absent)"
    gs = np.sqrt(max(g[2] * g[3], 0))
    cv2.rectangle(canvas, (0, 0), (a.panel, 40), (0, 0, 0), -1)
    cv2.putText(canvas, lab, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"GT {gs:.0f}px  view x{k:.1f}", (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    panels.append(canvas)

row = np.concatenate(panels, axis=1)
if a.title:
    bar = np.full((30, row.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, a.title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    row = np.concatenate([bar, row], axis=0)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
cv2.imwrite(a.out, row)
print(a.out, row.shape)

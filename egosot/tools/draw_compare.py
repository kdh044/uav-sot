"""두 트래커 결과 비교 (원본 프레임, 박스만, 텍스트 없음).
초록=GT, 빨강=기준 트래커, 파랑=개선 트래커.

usage: python tools/draw_compare.py <out_dir> <dataset> <seq> <base_dir> <ours_dir> <frame> [<frame> ...]
"""
import os, sys
import numpy as np, cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ORTrack'))
from lib.test.evaluation import get_dataset
from lib.test.utils.load_text import load_text

out, dname, sname, bdir, odir = sys.argv[1:6]
frames = [int(x) for x in sys.argv[6:]]
seq = [s for s in get_dataset(dname) if s.name == sname][0]
gt = np.asarray(seq.ground_truth_rect, float)
rd = lambda d: load_text(os.path.join(d, sname + '.txt'), delimiter=('\t', ','), dtype=np.float64)
base, ours = rd(bdir), rd(odir)
os.makedirs(out, exist_ok=True)


def rect(img, b, c, th):
    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[0] + b[2]), int(b[1] + b[3])), c, th)


for t in frames:
    img = cv2.imread(seq.frames[t])
    th = max(2, img.shape[1] // 600)
    if gt[t][2] > 0:
        rect(img, gt[t], (0, 220, 0), th)
    rect(img, base[t], (0, 0, 255), th)
    rect(img, ours[t], (255, 120, 0), th)
    cv2.imwrite(os.path.join(out, f'{sname}_{t:05d}.jpg'), img)
print(out)

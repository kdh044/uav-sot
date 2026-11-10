"""원본 프레임에 박스만 그려 저장 (텍스트 없음).
초록=GT, 빨강=예측, 노랑=직전 예측 기준 검색영역(4*sqrt(w*h)).

usage: python tools/draw_boxes.py <out_dir> <dataset> <seq> <result_dir> <frame> [<frame> ...]
"""
import os, sys
import numpy as np, cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ORTrack'))
from lib.test.evaluation import get_dataset
from lib.test.utils.load_text import load_text

out, dname, sname, rdir = sys.argv[1:5]
frames = [int(x) for x in sys.argv[5:]]
seq = [s for s in get_dataset(dname) if s.name == sname][0]
gt = np.asarray(seq.ground_truth_rect, float)
pr = load_text(os.path.join(rdir, sname + '.txt'), delimiter=('\t', ','), dtype=np.float64)
os.makedirs(out, exist_ok=True)
for t in frames:
    img = cv2.imread(seq.frames[t])
    th = max(2, img.shape[1] // 640)
    prev = pr[t - 1]
    side = 4 * np.sqrt(max(prev[2] * prev[3], 1))
    c = prev[:2] + prev[2:] / 2
    cv2.rectangle(img, (int(c[0] - side / 2), int(c[1] - side / 2)), (int(c[0] + side / 2), int(c[1] + side / 2)), (0, 220, 255), th)
    if gt[t][2] > 0:
        g = gt[t]; cv2.rectangle(img, (int(g[0]), int(g[1])), (int(g[0] + g[2]), int(g[1] + g[3])), (0, 220, 0), th)
    q = pr[t]; cv2.rectangle(img, (int(q[0]), int(q[1])), (int(q[0] + q[2]), int(q[1] + q[3])), (0, 0, 255), th)
    cv2.imwrite(os.path.join(out, f'{sname}_{t:05d}.jpg'), img)
print(out)

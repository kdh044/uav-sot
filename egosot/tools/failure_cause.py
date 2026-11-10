"""놓침(IoU=0 이 k프레임 이상 연속) 시작 시점의 원인 분류.

- out_of_search: 시작 프레임 GT 중심이 직전 예측 기준 검색영역(한 변 = 4*sqrt(w*h)) 밖
                 → 카메라/대상 이동이 검색영역을 벗어남 (motion 문제)
- in_search    : GT 가 검색영역 안에 있었는데 다른 곳에 박스 (distractor/drift, appearance 문제)
추가로 시작 시점 GT 크기, 직전 프레임 대비 GT 이동량(대상 크기 대비)을 기록.

usage: python tools/failure_cause.py <tag> <result_dir> <dataset> [<result_dir> <dataset> ...]
"""
import os, sys
import numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ORTrack'))
from lib.test.evaluation import get_dataset
from lib.test.analysis.extract_results import calc_iou_overlap
from lib.test.utils.load_text import load_text

K = 5
SEARCH_FACTOR = 4.0


def events(result_dir, dataset_name):
    out = []
    for seq in get_dataset(dataset_name):
        gt = np.asarray(seq.ground_truth_rect, dtype=np.float64)
        pr = load_text(os.path.join(result_dir, seq.name + '.txt'), delimiter=('\t', ','), dtype=np.float64)
        n = min(len(gt), len(pr))
        gt, pr = gt[:n], pr[:n]
        valid = (gt[:, 2] > 0) & (gt[:, 3] > 0)
        iou = calc_iou_overlap(torch.tensor(pr), torch.tensor(gt)).numpy()
        zero = (iou <= 0) & valid
        i = 1
        while i < n:
            if zero[i]:
                j = i
                while j < n and (zero[j] or not valid[j]):
                    j += 1
                if j - i >= K and i >= 1:
                    p = pr[i - 1]
                    side = SEARCH_FACTOR * np.sqrt(max(p[2] * p[3], 1e-6))
                    pc = p[:2] + p[2:] / 2
                    gc = gt[i, :2] + gt[i, 2:] / 2
                    inside = np.all(np.abs(gc - pc) <= side / 2)
                    size = np.sqrt(gt[i, 2] * gt[i, 3])
                    gprev = gt[i - 1] if valid[i - 1] else gt[i]
                    move = np.linalg.norm(gc - (gprev[:2] + gprev[2:] / 2)) / max(size, 1e-6)
                    out.append(dict(seq=seq.name, start=i, length=j - i, recovered=j < n,
                                    inside=bool(inside), size=size, move=move))
                i = j
            else:
                i += 1
    return out


if __name__ == '__main__':
    tag, args = sys.argv[1], sys.argv[2:]
    ev = []
    for k in range(0, len(args), 2):
        ev += events(args[k], args[k + 1])
    ev_in = [e for e in ev if e['inside']]
    ev_out = [e for e in ev if not e['inside']]
    tiny = [e for e in ev if e['size'] < 16]
    def rec(es): return 100 * np.mean([e['recovered'] for e in es]) if es else float('nan')
    print(f"{tag:14s} events {len(ev):4d} | in-search(distractor) {len(ev_in):3d} ({100*len(ev_in)/max(len(ev),1):4.1f}%) "
          f"rec {rec(ev_in):4.0f}% | out-of-search {len(ev_out):3d} ({100*len(ev_out)/max(len(ev),1):4.1f}%) "
          f"rec {rec(ev_out):4.0f}% | start<16px {len(tiny):3d} ({100*len(tiny)/max(len(ev),1):4.1f}%) "
          f"| median move/size in {np.median([e['move'] for e in ev_in]) if ev_in else 0:.2f} "
          f"out {np.median([e['move'] for e in ev_out]) if ev_out else 0:.2f}")
    np.save(f'analysis/failure_{tag}.npy', ev, allow_pickle=True)

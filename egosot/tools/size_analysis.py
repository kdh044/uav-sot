"""대상 크기(sqrt(w*h), GT 기준)별 성능 + 놓침/복구 분석.

usage: python tools/size_analysis.py analysis/<tag>_<dataset>.npy [...]
입력: tools/score.py --frames 로 저장한 프레임 단위 기록 (첫 프레임 제외, 무효 GT 제외)
"""
import sys
import numpy as np

BINS = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 1e9)]


def bin_table(a):
    print(f"{'size(px)':>10} {'frames':>8} {'share':>6} {'mIoU':>6} {'IoU>.5':>7} {'IoU=0':>6} {'P@20':>6}")
    for lo, hi in BINS:
        m = (a['size'] >= lo) & (a['size'] < hi)
        if not m.any():
            continue
        x = a[m]
        label = f"{lo}-{int(hi)}" if hi < 1e9 else f"{lo}+"
        print(f"{label:>10} {m.sum():8d} {100 * m.mean():5.1f}% "
              f"{x['iou'].mean():6.3f} {100 * (x['iou'] > .5).mean():6.1f}% "
              f"{100 * (x['iou'] <= 0).mean():5.1f}% {100 * (x['cerr'] <= 20).mean():5.1f}%")


def lost_recovery(a):
    """IoU=0 이 연속 k프레임 이상이면 '놓침'. 이후 IoU>0 으로 돌아오면 '복구'."""
    k = 5
    n_lost = n_rec = 0
    lost_len = []
    for s in np.unique(a['seq']):
        iou = a['iou'][a['seq'] == s]
        zero = iou <= 0
        i = 0
        while i < len(zero):
            if zero[i]:
                j = i
                while j < len(zero) and zero[j]:
                    j += 1
                if j - i >= k:
                    n_lost += 1
                    lost_len.append(j - i)
                    if j < len(zero):
                        n_rec += 1
                i = j
            else:
                i += 1
    seqs = len(np.unique(a['seq']))
    seq_with_loss = sum(((a['iou'][a['seq'] == s] <= 0).astype(int).sum() >= k) for s in np.unique(a['seq']))
    print(f"lost events(>= {k} frames IoU=0): {n_lost}  recovered: {n_rec} "
          f"({100 * n_rec / max(n_lost, 1):.0f}%)  median lost length: "
          f"{np.median(lost_len) if lost_len else 0:.0f}  seqs w/ loss: {seq_with_loss}/{seqs}")


if __name__ == '__main__':
    for f in sys.argv[1:]:
        a = np.load(f)
        a = a[a['valid']]
        print(f"\n## {f}")
        bin_table(a)
        lost_recovery(a)

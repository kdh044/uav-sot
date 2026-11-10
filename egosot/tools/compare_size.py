"""여러 트래커의 크기별 성능(표준 4종 합산) + BioDrone(보이는 프레임만) AUC 비교.

usage: python tools/compare_size.py <name>=<frames_tag> ...
  frames_tag: analysis/<frames_tag>_<dataset>.npy 접두어
"""
import sys
import numpy as np

B = [(0, 16), (16, 32), (32, 64), (64, 1e9)]
STD = ['dtb70', 'uavdt', 'visdrone2018', 'uav123']
TH = np.arange(0, 1.05, 0.05)


def auc_visible(a):
    r = [((a['iou'][a['seq'] == s][:, None] > TH).mean(0)).mean() for s in np.unique(a['seq'])]
    return 100 * np.mean(r)


print(f"{'model':16s} {'stdAUC':>7s} " + ' '.join(f"{('<16' if lo == 0 else f'{lo}+' if hi > 1e8 else f'{lo}-{int(hi)}'):>13s}" for lo, hi in B) + f" {'BioAUC':>7s}")
for arg in sys.argv[1:]:
    name, tag = arg.split('=')
    a = np.concatenate([np.load(f'analysis/{tag}_{d}.npy') for d in STD]); a = a[a['valid']]
    cells = []
    for lo, hi in B:
        x = a['iou'][(a['size'] >= lo) & (a['size'] < hi)]
        cells.append(f"{x.mean():.3f}/{100 * (x <= 0).mean():4.1f}%")
    try:
        b = np.load(f'analysis/{tag}_biodrone.npy'); bio = f"{auc_visible(b[b['valid']]):7.1f}"
    except FileNotFoundError:
        bio = f"{'-':>7s}"
    print(f"{name:16s} {auc_visible(a):7.1f} " + ' '.join(f"{c:>13s}" for c in cells) + f" {bio}")

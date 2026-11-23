"""대표 세트 AUC: ORTrack / Look Twice(41 자동 최소박스, 42 두 위치) / 원본 OSTrack-256 CE(51xx)."""
import os, json, numpy as np
from score_subset import O, RT, man, base, order, iou, load, auc
OS = '/home/airlab/user/kdh/SOT/OSTrack/output/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300'
print(f'{"데이터":13s}{"n":>3s} {"ORTrack":>8s} {"LT41":>7s} {"LT42":>7s} {"OSTrack":>8s}')
tot = [[] for _ in range(4)]
for k in order:
    m = man[k]; ds = m['src_ds']; i = order.index(k)
    cols = [(RT + base[k], ds), (RT + f'_41{i:02d}', 'biodrone'), (RT + f'_42{i:02d}', 'biodrone'), (OS + f'_51{i:02d}', 'biodrone')]
    sc = [[] for _ in cols]; n = 0
    for s in m['seqs']:
        gt = np.loadtxt(f'{O}/{k}/anno/{s}.txt', delimiter=',', ndmin=2); v = gt[1:, 2] > 0
        if v.sum() == 0 or gt[0, 2] <= 0: continue  # 첫 프레임 GT 없음 → 초기화 불가
        n += 1
        for j, (c, d) in enumerate(cols):
            r = load(c, d, s)
            if r is None and os.path.exists(f'{c}/{s}.txt'): r = np.loadtxt(f'{c}/{s}.txt', delimiter='\t' if '\t' in open(f'{c}/{s}.txt').readline() else ',', ndmin=2)
            if r is not None: sc[j].append(auc(iou(r[1:len(gt)], gt[1:])[v]))
    cell = lambda j: f'{100*np.mean(sc[j]):.2f}' if len(sc[j]) == n else f'({len(sc[j])}/{n})'
    for j in range(4):
        if len(sc[j]) == n: tot[j].append(100*np.mean(sc[j]))
    print(f'{k:13s}{n:3d} {cell(0):>8s} {cell(1):>7s} {cell(2):>7s} {cell(3):>8s}', flush=True)
print(f'{"평균(완료분)":13s}    ' + ' '.join(f'{np.mean(t):8.2f}({len(t)})' for t in tot))

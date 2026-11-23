"""대표 세트 AUC: ORTrack / v4(전체 실행에서 같은 시퀀스) / 핵심판 41(자동 최소박스) / 42(두 위치, 10px) / 43(두 위치+피드백, 10px)."""
import os, json, numpy as np
O = '/home/airlab/user/kdh/SOT/analysis/subset_20261113'; RT = '/home/airlab/user/kdh/SOT/ORTrack/output/test/tracking_results/ortrack/deit_tiny_patch16_224'
man = json.load(open(f'{O}/manifest.json'))
base = {'vetra': '_3610', 'birdsai': '_3200', 'biodrone': '_3600', 'sv248s': '_3601', 'satsot': '_3602', 'viso': '_3603', 'uav123': '', 'dtb70': '', 'uavdt': '', 'visdrone2018': ''}
v4 = {'sv248s': '3900', 'biodrone': '3901', 'birdsai': '3902', 'satsot': '3903', 'viso': '3904', 'uav123': '3905', 'dtb70': '3906', 'uavdt': '3907', 'visdrone2018': '3908', 'vetra': '3909'}
order = ['vetra', 'birdsai', 'biodrone', 'uav123', 'dtb70', 'uavdt', 'visdrone2018', 'sv248s', 'satsot', 'viso']
def iou(a, b):
    x1=np.maximum(a[:,0],b[:,0]);y1=np.maximum(a[:,1],b[:,1]);x2=np.minimum(a[:,0]+a[:,2],b[:,0]+b[:,2]);y2=np.minimum(a[:,1]+a[:,3],b[:,1]+b[:,3])
    i=np.clip(x2-x1,0,None)*np.clip(y2-y1,0,None);return i/(a[:,2]*a[:,3]+b[:,2]*b[:,3]-i+1e-9)
def load(d, ds, n):
    for p in [f'{d}/{ds}/result/{n.replace("frame_", "ORTrack_")}.txt', f'{d}/{ds}/{n}.txt', f'{d}/biodrone/result/{n.replace("frame_", "ORTrack_")}.txt']:
        if os.path.exists(p): return np.loadtxt(p, delimiter='\t' if '\t' in open(p).readline() else ',', ndmin=2)
def auc(u): return np.mean([(u > t).mean() for t in np.linspace(0, 1, 21)])
print(f'{"데이터":13s}{"n":>3s} {"ORTrack":>8s} {"v4":>7s} {"41자동":>7s} {"42두위치":>8s} {"43+피드백":>9s}')
for k in order:
    m = man[k]; ds = m['src_ds']; cols = [RT + base[k], RT + '_' + v4[k], RT + f'_41{order.index(k):02d}', RT + f'_42{order.index(k):02d}', RT + f'_43{order.index(k):02d}']
    dss = [ds, ds, 'biodrone', 'biodrone', 'biodrone']
    sc = [[] for _ in cols]; n = 0
    for s in m['seqs']:
        gt = np.loadtxt(f'{O}/{k}/anno/{s}.txt', delimiter=',', ndmin=2); v = gt[1:, 2] > 0
        if v.sum() == 0: continue
        rs = [load(c, d, s) for c, d in zip(cols, dss)]
        for j, r in enumerate(rs):
            if r is not None: sc[j].append(auc(iou(r[1:len(gt)], gt[1:])[v]))
        n += 1
    b = 100 * np.mean(sc[0]) if sc[0] else float('nan')
    cell = lambda j: (f'{100*np.mean(sc[j])-b:+.2f}' if len(sc[j]) == n else f'({len(sc[j])}/{n})')
    print(f'{k:13s}{n:3d} {b:8.2f} {cell(1):>7s} {cell(2):>7s} {cell(3):>8s} {cell(4):>9s}', flush=True)

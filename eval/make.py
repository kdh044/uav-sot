"""대표 세트: 데이터마다 이름순 정렬 후 고른 간격으로 최대 30개 시퀀스 → BioDrone 형식 뷰(원래 시퀀스 이름 유지)."""
import os, sys, json, importlib, numpy as np
sys.path.insert(0, '/home/airlab/user/kdh/SOT/ORTrack')
A = '/home/airlab/user/kdh/SOT/analysis'; O = f'{A}/subset_20261113'
src = {'vetra': ('biodrone', f'{A}/vetra_view_20261113/view'), 'birdsai': ('biodrone', f'{A}/birdsai_view_20261113/view'),
       'biodrone': ('biodrone', f'{A}/vd_view_20261113/biodrone_evalall'), 'uav123': ('uav123', None), 'dtb70': ('dtb70', None),
       'uavdt': ('uavdt', None), 'visdrone2018': ('visdrone2018', None), 'sv248s': ('biodrone', f'{A}/sat_view_20261113/sv248s'),
       'satsot': ('biodrone', f'{A}/sat_view_20261113/satsot'), 'viso': ('biodrone', f'{A}/sat_view_20261113/viso')}
from lib.test.evaluation import get_dataset
man = {}
for k, (ds, view) in src.items():
    if view: os.environ['SOT_BIODRONE'] = view
    import lib.test.evaluation.local as L; importlib.reload(L)
    S = sorted(get_dataset(ds), key=lambda q: q.name)
    pick = [S[int(i)] for i in np.linspace(0, len(S) - 1, min(30, len(S)))]
    d = f'{O}/{k}'; os.makedirs(f'{d}/sequences', exist_ok=True); os.makedirs(f'{d}/anno', exist_ok=True)
    for q in pick:
        sd = f'{d}/sequences/{q.name}'; os.makedirs(sd, exist_ok=True)
        for i, f in enumerate(q.frames):
            t = f'{sd}/{i + 1:08d}.jpg'
            if not os.path.exists(t): os.symlink(os.path.realpath(f), t)
        np.savetxt(f'{d}/anno/{q.name}.txt', np.asarray(q.ground_truth_rect, float), fmt='%.2f', delimiter=',')
    man[k] = dict(src_ds=ds, n=len(pick), seqs=[q.name for q in pick], frames=int(sum(len(q.frames) for q in pick)))
    print(k, len(pick), man[k]['frames'], flush=True)
json.dump(man, open(f'{O}/manifest.json', 'w'), indent=1)

"""주 지표 CSV 생성: 모델 x 데이터셋 별 Precision@20 / Success AUC (+BioDrone 보조 지표).

- 표준 4종: tools/score.py 와 동일 규칙 (전체 프레임, ORTrack analysis 규칙)
- BioDrone val: 대상이 보이는 프레임만 (시퀀스별 평균) — 공식 BioDrone toolkit 과 다름
- 놓침 이벤트/영역밖 비율: tools/failure_cause.py 결과 (IoU=0 이 5프레임 이상)
출력: analysis/main_metrics.csv
"""
import csv, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
os.chdir(ROOT)
from score import score

R = 'ORTrack/output/test/tracking_results/ortrack'
A = 'AVTrack/output/test/tracking_results/avtrack'
S = 'SGLATrack/output/test/tracking_results/sglatrack/deit_distilled'
L = 'LETrack/output/test/tracking_results/letrack/baseline_WOCE_298'
O = 'OSTrack/output/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300'

# name, venue, latency_ms(GPU, crop·전처리 포함), {dataset: result_dir}, biodrone frames npy, failure tag (biodrone), failure tag (std)
MODELS = [
    ('ORTrack-DeiT', 'CVPR25', 3.25, {d: f'{R}/deit_tiny_patch16_224/{d}' for d in ['dtb70', 'uavdt', 'visdrone2018', 'uav123']},
     'fp32_deit_tiny_patch16_224_biodrone', 'ORTrack-DeiT-bio', 'ORTrack-DeiT'),
    ('ORTrack-D', 'CVPR25', 2.24, {d: f'{R}/deit_tiny_distilled_patch16_224/{d}' for d in ['dtb70', 'uavdt', 'visdrone2018', 'uav123']},
     'ortrack_d_biodrone', 'ortrack_d-bio', 'ORTrack-D'),
    ('AVTrack-DeiT', 'ICML24', 3.11, {d: f'{A}/deit_tiny_patch16_224/{d}' for d in ['dtb70', 'uavdt', 'visdrone2018', 'uav123']},
     'avtrack_deit_biodrone', 'avtrack_deit-bio', 'AVTrack-DeiT'),
    ('AVTrack-D', 'ICML24', 2.06, {d: f'{A}/deit_tiny_distilled_patch16_224/{d}' for d in ['dtb70', 'uavdt', 'visdrone2018', 'uav123']},
     'avtrack_d_biodrone', 'avtrack_d-bio', 'AVTrack-D'),
    ('SGLATrack', 'CVPR25', 2.59, {'dtb70': f'{S}/dtb70', 'uavdt': f'{S}/uavdt', 'visdrone2018': f'{S}/visdrone', 'uav123': f'{S}/uav123'},
     'sgla_biodrone', 'sgla-bio', 'SGLATrack'),
    ('LETrack', 'CVPR26', 5.76, {'dtb70': f'{L}/dtb70', 'uavdt': f'{L}/uavdt', 'visdrone2018': f'{L}/visdrone', 'uav123': f'{L}/uav123'},
     'letrack_biodrone', 'letrack-bio', 'LETrack'),
    ('OSTrack-256 (ViT-B)', 'ECCV22', 7.20, {'uav123': O},
     'teacher_ost256_biodrone', 'OST256-bio', None),
    ('ORTrack-DeiT +CMC', 'ours (no retrain)', None, {}, 'cmc_biodrone', 'ORTrack-cmc-bio', None),
    ('ORTrack-DeiT +CMC+CV', 'ours (no retrain)', None, {}, 'cmccv_biodrone', 'ORTrack-cmccv-bio', None),
    ('ORTrack-DeiT +CMC (std)', 'ours (no retrain)', 8.04, {d: f'{R}/deit_tiny_patch16_224_801/{d}' for d in ['dtb70', 'uavdt', 'visdrone2018', 'uav123']},
     'cmc_biodrone', 'ORTrack-cmc-bio', None),
    ('ORTrack-DeiT drone-FT', 'ours', 2.97, {d: f'{R}/deit_tiny_patch16_224_uavft/{d}' for d in ['dtb70', 'uavdt', 'visdrone2018']} | {'uav123': f'{R}/deit_tiny_patch16_224_uavft_500/uav123'},
     'ft_val_nocmc', 'FT-bio', None),
    ('ORTrack-DeiT drone-FT +CMC (final)', 'ours', 8.04, {d: f'{R}/deit_tiny_patch16_224_uavft_801/{d}' for d in ['dtb70', 'uavdt', 'visdrone2018', 'uav123']},
     'ftcmc_val_biodrone', 'FTCMC-bio', None),
    ('ORTrack-DeiT +size-adaptive crop', 'ours (no retrain)', None,
     {d: f'{R}/deit_tiny_patch16_224_024/{d}' for d in ['dtb70', 'uavdt', 'visdrone2018', 'uav123']}, 'ortrack_ad24_biodrone', None, None),
]
DS = ['uav123', 'dtb70', 'uavdt', 'visdrone2018']
TH = np.arange(0, 1.05, 0.05)


def bio(npy):
    p = f'analysis/{npy}.npy'
    if not os.path.exists(p):
        return None, None
    a = np.load(p); a = a[a['valid']]
    seqs = np.unique(a['seq'])
    auc = np.mean([((a['iou'][a['seq'] == s][:, None] > TH).mean(0)).mean() for s in seqs])
    pr = np.mean([(a['cerr'][a['seq'] == s] <= 20).mean() for s in seqs])
    return 100 * pr, 100 * auc


def fail(tag):
    if not tag or not os.path.exists(f'analysis/failure_{tag}.npy'):
        return None, None
    ev = list(np.load(f'analysis/failure_{tag}.npy', allow_pickle=True))
    return len(ev), 100 * np.mean([not e['inside'] for e in ev])


rows = []
for name, venue, lat, dirs, bnpy, fbio, fstd in MODELS:
    r = {'model': name, 'venue': venue}
    aucs = []
    for d in DS:
        if d in dirs and os.path.isdir(dirs[d]):
            pr, au, _ = score(dirs[d], d)
            r[f'{d}_prec20'], r[f'{d}_auc'] = round(pr, 1), round(au, 1)
            aucs.append(au)
    if len(aucs) == 4:
        r['std4_mean_auc'] = round(np.mean(aucs), 1)
    bp, ba = bio(bnpy)
    if ba is not None:
        r['biodrone_val_prec20'], r['biodrone_val_auc'] = round(bp, 1), round(ba, 1)
    n, out = fail(fbio)
    if n is not None:
        r['biodrone_lost_events'], r['biodrone_out_of_roi_pct'] = n, round(out, 1)
    n, out = fail(fstd)
    if n is not None:
        r['std4_lost_events'], r['std4_out_of_roi_pct'] = n, round(out, 1)
    if lat:
        r['gpu_latency_ms'], r['gpu_fps'] = lat, round(1000 / lat)
    rows.append(r)
    print(r, flush=True)

cols = ['model', 'venue'] + [f'{d}_{m}' for d in DS for m in ('prec20', 'auc')] + \
       ['std4_mean_auc', 'biodrone_val_prec20', 'biodrone_val_auc', 'biodrone_lost_events', 'biodrone_out_of_roi_pct',
        'std4_lost_events', 'std4_out_of_roi_pct', 'gpu_latency_ms', 'gpu_fps']
with open('analysis/main_metrics.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
print('analysis/main_metrics.csv')

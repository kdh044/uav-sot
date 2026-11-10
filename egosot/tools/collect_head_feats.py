"""B1: 고정된 ORTrack-DeiT 를 돌리며 프레임별 '놓침 감지 헤드' 학습용 특징 수집.

network.forward 를 감싸 out_dict(score_map, backbone_feat)만 읽는다 → 추적 결과는 원본과 동일.
특징: score map 통계, 박스 변화, 예측 중심 search 토큰(192), 템플릿 중심 토큰과의 cos 등.
라벨: IoU(pred, GT), lost(IoU==0), absent(GT 무효).

usage: python tools/collect_head_feats.py <kind:bio|vis> <root> <out.npz> [--shard i/n] [--max_seqs N] [--check_dir DIR]
"""
import argparse, os, sys
import numpy as np, torch, cv2

p = argparse.ArgumentParser()
p.add_argument('kind'); p.add_argument('root'); p.add_argument('out')
p.add_argument('--shard', default='0/1'); p.add_argument('--max_seqs', type=int, default=0)
p.add_argument('--check_dir', default=None, help='원본 결과 폴더와 예측 비교 (BioDrone: ORTrack_XXX.txt)')
a = p.parse_args()

REPO = '/home/airlab/user/kdh/SOT/ORTrack'
sys.path.insert(0, REPO); os.chdir(os.path.join(REPO, 'tracking'))
from lib.test.evaluation.tracker import Tracker

torch.set_num_threads(1)


def list_seqs(kind, root):
    seq_dir = os.path.join(root, 'sequences')
    out = []
    for s in sorted(os.listdir(seq_dir)):
        if kind == 'bio':
            anno = os.path.join(root, 'anno', s + '.txt')
        else:
            anno = os.path.join(root, 'annotations', s + '.txt')
        fs = sorted(f for f in os.listdir(os.path.join(seq_dir, s)) if f.endswith('.jpg'))
        gt = np.loadtxt(anno, delimiter=',', ndmin=2)
        out.append((s, [os.path.join(seq_dir, s, f) for f in fs], gt))
    return out


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0., min(ax2, bx2) - max(a[0], b[0])); ih = max(0., min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih; u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.


seqs = list_seqs(a.kind, a.root)
i, n = map(int, a.shard.split('/'))
seqs = seqs[i::n]
if a.max_seqs:
    seqs = seqs[:a.max_seqs]

trk = Tracker('ortrack', 'deit_tiny_patch16_224', 'uav123')
params = trk.get_parameters(); params.debug = 0
tracker = trk.create_tracker(params)
stash = {}
orig = tracker.network.forward
def hooked(*args, **kw):
    o = orig(*args, **kw)
    stash['o'] = o
    return o
tracker.network.forward = hooked

FS = 16  # search feature map 16x16
T_SZ = 8  # template token grid 8x8
tc = [r * T_SZ + c for r in range(2, 6) for c in range(2, 6)]  # 템플릿 중심 4x4 = 대상 영역

rows_stat, rows_tok, rows_lab, rows_seq = [], [], [], []
for s, frames, gt in seqs:
    n_f = min(len(frames), len(gt))
    img0 = cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2RGB)
    init = list(gt[0])
    if init[2] <= 0 or init[3] <= 0:
        continue
    tracker.initialize(img0, {'init_bbox': init})
    prev = list(init); ref_tok = None; preds = [init]
    for k in range(1, n_f):
        img = cv2.cvtColor(cv2.imread(frames[k]), cv2.COLOR_BGR2RGB)
        box = tracker.track(img)['target_bbox']
        preds.append(box)
        o = stash['o']
        sm = o['score_map'][0, 0].float()                       # 16x16
        resp = (tracker.output_window[0, 0] * sm) if tracker.output_window.dim() == 4 else tracker.output_window * sm
        resp = resp.reshape(FS, FS)
        idx = int(torch.argmax(resp)); r, c = divmod(idx, FS)
        smf = sm.flatten(); smax = float(sm.max()); smean = float(smf.mean())
        pn = smf / (smf.sum() + 1e-8); ent = float(-(pn * (pn + 1e-12).log()).sum())
        sup = sm.clone(); sup[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = 0
        idx2 = int(torch.argmax(sup)); r2, c2 = divmod(idx2, FS); s2 = float(sup.max())
        d12 = float(np.hypot(r - r2, c - c2))
        sharp = float(sm[max(0, r - 1):r + 2, max(0, c - 1):c + 2].sum() / (smf.sum() + 1e-8))
        feat = o['backbone_feat']; feat = feat[-1] if isinstance(feat, (list, tuple)) else feat
        feat = feat[0].float()                                   # (64+256, 192)
        ztok = feat[:T_SZ * T_SZ][tc].mean(0)
        xtok = feat[T_SZ * T_SZ:][idx]
        cos_z = float(torch.nn.functional.cosine_similarity(xtok, ztok, dim=0))
        if ref_tok is None:
            ref_tok = xtok.clone()
        cos_ref = float(torch.nn.functional.cosine_similarity(xtok, ref_tok, dim=0))
        dsz = float(np.log((box[2] * box[3] + 1e-6) / (prev[2] * prev[3] + 1e-6)))
        dar = float(np.log((box[2] / max(box[3], 1e-6)) / (prev[2] / max(prev[3], 1e-6) + 1e-9) + 1e-9))
        mv = float(np.hypot(box[0] + box[2] / 2 - prev[0] - prev[2] / 2, box[1] + box[3] / 2 - prev[1] - prev[3] / 2)
                   / max(np.sqrt(prev[2] * prev[3]), 1e-6))
        size = float(np.sqrt(box[2] * box[3]))
        prev = list(box)
        g = gt[k]
        absent = not (g[2] > 0 and g[3] > 0)
        iu = 0. if absent else iou(box, g)
        rows_stat.append([smax, float(resp.max()), s2, s2 / (smax + 1e-8), d12, ent, smean, sharp,
                          cos_z, cos_ref, dsz, dar, mv, np.log(size + 1)])
        rows_tok.append(xtok.cpu().numpy().astype(np.float16))
        rows_lab.append([iu, float(iu <= 0 and not absent), float(absent), k])
        rows_seq.append(s)
    if a.check_dir:
        sid = s.split('_')[-1]
        ref = np.loadtxt(os.path.join(a.check_dir, f'ORTrack_{sid}.txt'), delimiter=',')
        dif = np.abs(np.array(preds[1:]) - ref[1:len(preds)]).max()
        print(f'CHECK {s} max abs diff vs original = {dif:.4f}', flush=True)
    print(s, n_f, flush=True)

STAT_NAMES = ['smax', 'rmax', 's2', 's2_ratio', 'd12', 'entropy', 'smean', 'sharp',
              'cos_tmpl', 'cos_ref', 'dlog_area', 'dlog_ar', 'move', 'log_size']
np.savez_compressed(a.out, stat=np.array(rows_stat, np.float32), tok=np.array(rows_tok),
                    lab=np.array(rows_lab, np.float32), seq=np.array(rows_seq), names=np.array(STAT_NAMES))
print('saved', a.out, len(rows_stat))

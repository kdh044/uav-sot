"""fake-quant 변형별 요약: UAV123 Prec/AUC(전체 프레임), BioDrone AUC(보이는 프레임만), <16px mIoU/IoU=0."""
import numpy as np, sys
th = np.arange(0, 1.05, 0.05)
def vis_auc(a):
    a = a[a['valid']]; r = []; p = []
    for s in np.unique(a['seq']):
        x = a[a['seq'] == s]; r.append(((x['iou'][:, None] > th).mean(0)).mean()); p.append((x['cerr'] <= 20).mean())
    return 100 * np.mean(p), 100 * np.mean(r)
def tiny(a):
    a = a[a['valid'] & (a['size'] < 16)]
    return a['iou'].mean(), 100 * (a['iou'] <= 0).mean(), len(a)
for v in sys.argv[1:]:
    u = np.load(f'analysis/fq/{v}_uav123.npy'); b = np.load(f'analysis/fq/{v}_biodrone.npy')
    both = np.concatenate([u, b]); tm, tl, tn = tiny(both)
    print(f"{v:15s} UAV123 vis Prec/AUC {vis_auc(u)[0]:5.1f}/{vis_auc(u)[1]:5.1f} | BioDrone vis AUC {vis_auc(b)[1]:5.1f} | <16px (n={tn}) mIoU {tm:.3f} IoU=0 {tl:4.1f}%")

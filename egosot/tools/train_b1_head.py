"""B1: 놓침 감지 헤드 학습·평가 (ORTrack-DeiT 고정, 특징은 tools/collect_head_feats.py).

학습: BioDrone train 80개 (시퀀스 단위 80/20 로 모델 선택)
평가: BioDrone val. 양성 = 대상이 보이는데 IoU==0 인 프레임. 대상 없음(absent) 프레임은 제외.
비교: 원 신뢰도(score max) / 로지스틱(통계 특징) / 로지스틱(통계+시간) / MLP(통계+시간+토큰)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

A = '/home/airlab/user/kdh/SOT/analysis/'
W = 5  # 시간 특징 창


def load(files):
    d = [np.load(A + f) for f in files]
    stat = np.concatenate([x['stat'] for x in d]); tok = np.concatenate([x['tok'] for x in d]).astype(np.float32)
    lab = np.concatenate([x['lab'] for x in d]); seq = np.concatenate([x['seq'] for x in d])
    names = list(d[0]['names'])
    return stat, tok, lab, seq, names


def temporal(stat, seq, names):
    """인과적(과거만) 시간 특징: 최근 W프레임 smax 평균·최소, cos_ref 평균, smax 변화량."""
    i_s, i_c = names.index('smax'), names.index('cos_ref')
    out = np.zeros((len(stat), 4), np.float32)
    start = 0
    for k in range(1, len(seq) + 1):
        if k == len(seq) or seq[k] != seq[start]:
            s = stat[start:k, i_s]; c = stat[start:k, i_c]
            for j in range(k - start):
                lo = max(0, j - W + 1)
                out[start + j] = [s[lo:j + 1].mean(), s[lo:j + 1].min(), c[lo:j + 1].mean(),
                                  s[j] - (s[j - 1] if j else s[j])]
            start = k
    return out


def rate_at_fpr(y, score, fpr_t):
    fpr, tpr, thr = roc_curve(y, score)
    j = np.searchsorted(fpr, fpr_t, side='right') - 1
    return tpr[j], thr[j]


def event_flag(lab, seq, score, thr, k=5, within=5):
    """대상이 보이는데 IoU==0 이 k프레임 이상 이어진 이벤트 중, 시작 후 within 프레임 안에 경보가 난 비율."""
    n_ev = n_hit = 0; start = 0
    lost = (lab[:, 1] > 0)
    for e in range(1, len(seq) + 1):
        if e == len(seq) or seq[e] != seq[start]:
            L = lost[start:e]; S = score[start:e]; i = 0
            while i < len(L):
                if L[i]:
                    j = i
                    while j < len(L) and L[j]:
                        j += 1
                    if j - i >= k:
                        n_ev += 1; n_hit += bool((S[i:i + within] >= thr).any())
                    i = j
                else:
                    i += 1
            start = e
    return n_hit / max(n_ev, 1), n_ev


tr = load(['b1_bio_train_0.npz', 'b1_bio_train_1.npz'])
va = load(['b1_bio_val.npz'])
names = tr[4]


def build(d):
    stat, tok, lab, seq, _ = d
    keep = lab[:, 2] == 0  # absent 제외
    tmp = temporal(stat, seq, names)
    return stat[keep], tmp[keep], tok[keep], lab[keep], seq[keep]


Xs, Xt, Xk, Ytr, Str = build(tr)
Vs, Vt, Vk, Yva, Sva = build(va)
y, yv = Ytr[:, 1], Yva[:, 1]
print(f'train frames {len(y)} (lost {y.mean():.3f}), val frames {len(yv)} (lost {yv.mean():.3f})')

# 시퀀스 단위 80/20 (모델 선택용)
useq = np.unique(Str); rng = np.random.RandomState(0); rng.shuffle(useq)
hold = np.isin(Str, useq[:max(1, len(useq) // 5)])

res = {}
res['score max (원 신뢰도)'] = -Vs[:, names.index('smax')]

sets = {'로지스틱(통계)': (Xs, Vs), '로지스틱(통계+시간)': (np.hstack([Xs, Xt]), np.hstack([Vs, Vt]))}
for nm, (X, V) in sets.items():
    sc = StandardScaler().fit(X[~hold])
    best = None
    for C in [0.01, 0.1, 1.0]:
        m = LogisticRegression(C=C, max_iter=2000, class_weight='balanced').fit(sc.transform(X[~hold]), y[~hold])
        a = roc_auc_score(y[hold], m.decision_function(sc.transform(X[hold])))
        if best is None or a > best[0]:
            best = (a, C)
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=best[1], max_iter=2000, class_weight='balanced').fit(sc.transform(X), y)
    res[nm] = m.decision_function(sc.transform(V))
    if nm == '로지스틱(통계+시간)':
        coef = m.coef_[0]; fn = names + ['smax_mean5', 'smax_min5', 'cosref_mean5', 'dsmax']
        top = np.argsort(-np.abs(coef))[:8]
        print('logistic top features:', ', '.join(f'{fn[i]}({coef[i]:+.2f})' for i in top))

X = np.hstack([Xs, Xt, Xk]); V = np.hstack([Vs, Vt, Vk])
sc = StandardScaler().fit(X)
# 클래스 균형: 양성 오버샘플
pos = np.where(y > 0)[0]; neg = np.where(y == 0)[0]
idx = np.concatenate([neg, rng.choice(pos, len(neg) // 3, replace=True)])
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True, max_iter=60, random_state=0)
mlp.fit(sc.transform(X[idx]), y[idx])
res['MLP(통계+시간+토큰)'] = mlp.predict_proba(sc.transform(V))[:, 1]

# IoU 회귀(참고)
reg = MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True, max_iter=60, random_state=0)
sub = rng.choice(len(X), min(len(X), 200000), replace=False)
reg.fit(sc.transform(X[sub]), Ytr[sub, 0])
pred_iou = reg.predict(sc.transform(V))
print(f'IoU 회귀 상관: MLP {np.corrcoef(pred_iou, Yva[:, 0])[0, 1]:.3f} vs smax {np.corrcoef(Vs[:, 0], Yva[:, 0])[0, 1]:.3f}')

print(f"\n{'방법':24s} {'AUROC':>6s} {'탐지@1%FA':>9s} {'탐지@5%FA':>9s} {'이벤트5f내(@5%FA)':>16s}")
out = {}
for nm, s in res.items():
    au = roc_auc_score(yv, s); t1, _ = rate_at_fpr(yv, s, 0.01); t5, th5 = rate_at_fpr(yv, s, 0.05)
    ev, n_ev = event_flag(Yva, Sva, s, th5)
    out[nm] = (au, t1, t5, ev)
    print(f'{nm:24s} {au:6.3f} {100 * t1:8.1f}% {100 * t5:8.1f}% {100 * ev:14.1f}%  (events {n_ev})')
np.save(A + 'b1_results.npy', out, allow_pickle=True)

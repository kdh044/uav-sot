"""Hailo DFC INT8 보정용 (template, search) 쌍 생성 — 학습 분할에서만 (평가 누수 없음).

BioDrone train(튜닝 30개 제외) + VisDrone-SOT train 에서 무작위 프레임 쌍을 골라
ORTrack 과 같은 crop(template 2배→128, search 4배→256)을 uint8 NHWC 로 저장.
search 는 실제 추적 오차를 흉내 내도록 중심·크기에 작은 jitter.
정규화(mean/std)는 DFC 의 normalization 레이어(.alls)에서 처리하도록 원본 RGB 그대로 둠.

usage: python tools/make_hailo_calib.py <out.npz> [--n 1024]
"""
import argparse, glob, os, random, sys
import numpy as np, cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ORTrack'))
from lib.train.data.processing_utils import sample_target

p = argparse.ArgumentParser(); p.add_argument('out'); p.add_argument('--n', type=int, default=1024)
a = p.parse_args()
random.seed(0); np.random.seed(0)
D = '/home/airlab/user/kdh/datasets/sot'
tune = set(os.listdir(f'{D}/biodrone_tune/sequences'))
seqs = []
for s in sorted(os.listdir(f'{D}/biodrone_train/sequences')):
    if s in tune:
        continue
    fr = sorted(glob.glob(f'{D}/biodrone_train/sequences/{s}/*.jpg'), key=lambda f: int(os.path.basename(f)[3:-4]))
    seqs.append((fr, np.loadtxt(f'{D}/biodrone_train/anno/{s}.txt', delimiter=',')))
for s in sorted(os.listdir(f'{D}/visdrone_sot_trainval/train/sequences')):
    fr = sorted(glob.glob(f'{D}/visdrone_sot_trainval/train/sequences/{s}/*.jpg'))
    seqs.append((fr, np.loadtxt(f'{D}/visdrone_sot_trainval/train/annotations/{s}.txt', delimiter=',')))

Z, X = [], []
while len(Z) < a.n:
    fr, gt = random.choice(seqs)
    n = min(len(fr), len(gt))
    ok = np.where((gt[:n, 2] > 2) & (gt[:n, 3] > 2))[0]
    if len(ok) < 2:
        continue
    i = int(random.choice(ok)); j = int(random.choice(ok[(ok > i) & (ok < i + 100)])) if ((ok > i) & (ok < i + 100)).any() else i
    zi = cv2.cvtColor(cv2.imread(fr[i]), cv2.COLOR_BGR2RGB); xi = cv2.cvtColor(cv2.imread(fr[j]), cv2.COLOR_BGR2RGB)
    zb = gt[i].copy(); xb = gt[j].copy()
    sz = np.sqrt(xb[2] * xb[3])
    xb[:2] += np.random.uniform(-0.5, 0.5, 2) * sz * 0.5      # 중심 jitter
    xb[2:] *= np.exp(np.random.uniform(-0.15, 0.15, 2))       # 크기 jitter
    zp, _, _ = sample_target(zi, list(zb), 2.0, output_sz=128)
    xp, _, _ = sample_target(xi, list(xb), 4.0, output_sz=256)
    Z.append(zp.astype(np.uint8)); X.append(xp.astype(np.uint8))
np.savez_compressed(a.out, template=np.stack(Z), search=np.stack(X))
print(a.out, np.stack(Z).shape, np.stack(X).shape)

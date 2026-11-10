"""놓침 전후 신뢰도(점수맵 최댓값) 추적.

ORTrack 계열 tracker.track() 를 그대로 돌리면서 network 출력 score_map 의 최댓값을
프레임마다 기록. 놓침 이벤트(analysis/failure_<tag>.npy)가 있는 시퀀스만 실행.

usage: python tools/score_trace.py <repo> <tracker> <config> <failure_tag> <out.npz>
"""
import os, sys
import numpy as np, torch, cv2

repo, name, config, ftag, out = sys.argv[1:6]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
events = list(np.load(os.path.join(ROOT, 'analysis', f'failure_{ftag}.npy'), allow_pickle=True))
seqs_needed = sorted({e['seq'] for e in events})

sys.path.insert(0, os.path.abspath(repo))
os.chdir(os.path.join(os.path.abspath(repo), 'tracking'))
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker

trk = Tracker(name, config, 'uav123')
params = trk.get_parameters(); params.debug = 0
tracker = trk.create_tracker(params)

# network forward 를 감싸 score_map 최댓값 기록
last = {}
orig_forward = tracker.network.forward
def hooked(*a, **k):
    o = orig_forward(*a, **k)
    last['score'] = float(o['score_map'].max())
    return o
tracker.network.forward = hooked

result = {}
for dname in ['dtb70', 'uavdt', 'visdrone2018', 'uav123']:
    for seq in get_dataset(dname):
        if seq.name not in seqs_needed:
            continue
        frames = seq.frames
        tracker.initialize(cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2RGB),
                           {'init_bbox': list(seq.ground_truth_rect[0])})
        scores, boxes = [np.nan], [list(seq.ground_truth_rect[0])]
        for f in frames[1:]:
            o = tracker.track(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB))
            scores.append(last['score']); boxes.append(o['target_bbox'])
        result[seq.name] = (np.array(scores), np.array(boxes))
        print(seq.name, len(scores), flush=True)
np.savez(os.path.join(ROOT, out), **{k: v[0] for k, v in result.items()})

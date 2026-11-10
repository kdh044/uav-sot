"""모든 트래커를 같은 방식으로 프레임당 지연 측정.

이미지를 미리 메모리에 올려두고(디스크 읽기 제외), tracker.track() 한 번을
cuda.synchronize 로 감싸 잰다. crop·전처리·네트워크·후처리 포함.

usage: python tools/bench_latency.py <repo> <tracker_name> <config> [--seq uav_car1_1] [--device cuda|cpu]
"""
import argparse, os, sys, time
import numpy as np, torch, cv2

p = argparse.ArgumentParser()
p.add_argument('repo'); p.add_argument('name'); p.add_argument('config')
p.add_argument('--seq', default='uav_car1_1')
p.add_argument('--n', type=int, default=500)
p.add_argument('--device', default='cuda')
p.add_argument('--threads', type=int, default=0, help='CPU 측정 시 torch 스레드 수')
a = p.parse_args()

repo = os.path.abspath(a.repo)
sys.path.insert(0, repo)
os.chdir(os.path.join(repo, 'tracking'))
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker

if a.threads:
    torch.set_num_threads(a.threads)

seq = [s for s in get_dataset('uav123') if s.name == a.seq][0]
frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in seq.frames[:a.n + 1]]

trk = Tracker(a.name, a.config, 'uav123')
params = trk.get_parameters()
params.debug = 0
tracker = trk.create_tracker(params)
if a.device == 'cpu':
    tracker.network = tracker.network.cpu()
    for k in [k for k in vars(tracker) if torch.is_tensor(getattr(tracker, k))]:
        setattr(tracker, k, getattr(tracker, k).cpu())

init = {'init_bbox': list(seq.ground_truth_rect[0])}
tracker.initialize(frames[0], init)
sync = torch.cuda.synchronize if a.device == 'cuda' else (lambda: None)
ts = []
for i, f in enumerate(frames[1:]):
    sync(); t0 = time.perf_counter()
    tracker.track(f)
    sync(); ts.append(time.perf_counter() - t0)
t = np.array(ts[20:])  # warmup 제외
print(f"{a.name}/{a.config} [{a.device}] n={len(t)} mean {1000 * t.mean():.2f} ms  "
      f"FPS {1 / t.mean():.1f}  p95 {1000 * np.percentile(t, 95):.2f} ms")

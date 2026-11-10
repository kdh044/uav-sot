"""ORTrack 계열 네트워크를 ONNX 로 내보내고 PyTorch 와 수치 비교 (Hailo DFC 입력 준비).

입력: template (1,3,128,128), search (1,3,256,256) — 정규화된 텐서
출력: score_map, size_map, offset_map (박스 디코딩·Hann 창은 CPU 후처리로 남김)

usage: python tools/export_onnx.py <config> <out.onnx> [--opset 17]
"""
import argparse, os, sys
import numpy as np, torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ORTrack')
sys.path.insert(0, ROOT)
os.chdir(os.path.join(ROOT, 'tracking'))
from lib.test.evaluation.tracker import Tracker

p = argparse.ArgumentParser()
p.add_argument('config'); p.add_argument('out'); p.add_argument('--opset', type=int, default=17)
a = p.parse_args()

trk = Tracker('ortrack', a.config, 'uav123')
params = trk.get_parameters(); params.debug = 0
net = trk.create_tracker(params).network.cpu().eval()
is_distill = params.cfg.MODEL['IS_DISTILL']


class Wrap(torch.nn.Module):
    def __init__(self, n):
        super().__init__(); self.n = n

    def forward(self, template, search):
        o = self.n.forward(template=template, search=search, is_distill=is_distill)
        return o['score_map'], o['size_map'], o['offset_map']


w = Wrap(net).eval()
z = torch.randn(1, 3, 128, 128); x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    ref = [t.numpy() for t in w(z, x)]
out = os.path.abspath(os.path.join('/home/airlab/user/kdh/SOT', a.out)) if not os.path.isabs(a.out) else a.out
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.onnx.export(w, (z, x), out, input_names=['template', 'search'],
                  output_names=['score_map', 'size_map', 'offset_map'], opset_version=a.opset, do_constant_folding=True,
                  dynamo=False)
import onnx, onnxruntime as ort
m = onnx.load(out); onnx.checker.check_model(m)
ops = sorted({n.op_type for n in m.graph.node})
sess = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
got = sess.run(None, {'template': z.numpy(), 'search': x.numpy()})
for name, r, g in zip(['score_map', 'size_map', 'offset_map'], ref, got):
    print(f'{name:10s} shape {g.shape}  max|diff| {np.abs(r - g).max():.2e}')
print('ops:', ', '.join(ops))
print('saved', out, f'{os.path.getsize(out) / 1e6:.1f} MB')

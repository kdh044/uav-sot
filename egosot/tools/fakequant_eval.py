"""ORTrack(-D) INT8 PTQ 모의 평가 (Hailo 없이 PyTorch fake-quant).

- 가중치: Linear/Conv 출력채널별 대칭 INT8
- 활성값: Linear/Conv 입력마다 텐서 단위 비대칭 UINT8 (zero point), 보정 범위 minmax 또는 percentile
- (선택) attention: q·k·v 와 softmax 출력도 텐서 단위 INT8 (fused SDPA 끔)
보정: VisDrone-SOT train 에서 뽑은 template/search 쌍 (평가셋과 분리)

usage:
  python tools/fakequant_eval.py calib  --cfg deit_tiny_distilled_patch16_224 --n 1000 --out analysis/fq_calib_D.pt
  python tools/fakequant_eval.py run    --cfg ... --calib analysis/fq_calib_D.pt --variant w8a8_pct \
         --dataset uav123 --out <result_dir> [--workers 4] [--seqs a,b,c] [--skip-blocks 0,1]
variant: fp32 | w8 | w8a8_mm | w8a8_pct | w8a8_pct_attn
"""
import argparse, os, sys, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, 'ORTrack')
CALIB_ROOT = '/home/airlab/user/kdh/datasets/sot/visdrone_sot_trainval/train'


def build_tracker(cfg):
    sys.path.insert(0, REPO)
    os.chdir(os.path.join(REPO, 'tracking'))
    from lib.test.evaluation.tracker import Tracker
    t = Tracker('ortrack', cfg, 'uav123')
    p = t.get_parameters(); p.debug = 0
    return t.create_tracker(p)


def quant_modules(net):
    return [(n, m) for n, m in net.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))]


def attn_modules(net):
    from lib.models.ortrack.vision_transformer import Attention
    return [(n, m) for n, m in net.named_modules() if isinstance(m, Attention)]


# ---------------------------------------------------------------- calibration
class Stat:
    def __init__(self): self.mn, self.mx, self.lo, self.hi, self.n = 1e9, -1e9, 0., 0., 0

    def add(self, x):
        x = x.detach().float().flatten()
        if x.numel() > 4_000_000:
            x = x[torch.randperm(x.numel(), device=x.device)[:4_000_000]]
        self.mn = min(self.mn, x.min().item()); self.mx = max(self.mx, x.max().item())
        q = torch.quantile(x, torch.tensor([1e-4, 1 - 1e-4], device=x.device))
        self.lo += q[0].item(); self.hi += q[1].item(); self.n += 1

    def ranges(self): return dict(mm=(self.mn, self.mx), pct=(self.lo / self.n, self.hi / self.n))


def calib_pairs(n, seed=0):
    from lib.train.data.processing_utils import sample_target
    rng = random.Random(seed)
    seqs = sorted(os.listdir(os.path.join(CALIB_ROOT, 'sequences')))
    out = []
    while len(out) < n:
        s = rng.choice(seqs)
        gt = np.loadtxt(os.path.join(CALIB_ROOT, 'annotations', s + '.txt'), delimiter=',')
        frames = sorted(os.listdir(os.path.join(CALIB_ROOT, 'sequences', s)))
        m = min(len(gt), len(frames))
        i = rng.randrange(0, m - 11); j = i + rng.randint(1, 10)
        if gt[i, 2] <= 1 or gt[j - 1, 2] <= 1 or gt[i, 3] <= 1 or gt[j - 1, 3] <= 1:
            continue
        rd = lambda k: cv2.cvtColor(cv2.imread(os.path.join(CALIB_ROOT, 'sequences', s, frames[k])), cv2.COLOR_BGR2RGB)
        z, _, _ = sample_target(rd(i), list(gt[i]), 2.0, output_sz=128)
        x, _, _ = sample_target(rd(j), list(gt[j - 1]), 4.0, output_sz=256)
        out.append((z, x))
    return out


def to_tensor(arrs):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    t = torch.tensor(np.stack(arrs)).cuda().float().permute(0, 3, 1, 2)
    return (t / 255. - mean) / std


def do_calib(a):
    tr = build_tracker(a.cfg); net = tr.network
    stats = {n: Stat() for n, _ in quant_modules(net)}
    hooks = [m.register_forward_pre_hook(lambda m, inp, n=n: stats[n].add(inp[0])) for n, m in quant_modules(net)]
    # attention 내부 q,k,v,softmax 통계
    astats = {}
    for n, m in attn_modules(net):
        for k in ['q', 'k', 'v', 'p']:
            astats[f'{n}.{k}'] = Stat()
        m.fused_attn = False
        m._fq_name = n
        m.forward = make_attn_forward(m, astats, None)
    pairs = calib_pairs(a.n)
    with torch.no_grad():
        for b in range(0, len(pairs), 16):
            zs, xs = zip(*pairs[b:b + 16])
            net(template=to_tensor(zs), search=to_tensor(xs), is_distill=tr.is_distill)
    for h in hooks: h.remove()
    torch.save(dict(lin={k: v.ranges() for k, v in stats.items()},
                    attn={k: v.ranges() for k, v in astats.items()}), os.path.join(ROOT, a.out))
    print('saved', a.out, len(stats), 'linear/conv,', len(astats), 'attn tensors')


# ---------------------------------------------------------------- fake quant
def fq_asym(x, lo, hi):
    lo, hi = min(lo, 0.), max(hi, 0.)
    s = max(hi - lo, 1e-8) / 255.
    zp = round(-lo / s)
    return (torch.clamp(torch.round(x / s) + zp, 0, 255) - zp) * s


def fq_weight(w):
    wf = w.flatten(1)
    s = wf.abs().amax(1).clamp(min=1e-8) / 127.
    shape = [-1] + [1] * (w.dim() - 1)
    return torch.clamp(torch.round(w / s.view(shape)), -127, 127) * s.view(shape)


def make_attn_forward(m, astats, aq):
    """Attention.forward 재정의. astats 가 있으면 통계 수집, aq(ranges) 가 있으면 fake-quant."""
    def fwd(x):
        B, N, C = x.shape
        qkv = m.qkv(x).reshape(B, N, 3, m.num_heads, m.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = m.q_norm(q), m.k_norm(k)
        n = m._fq_name
        if astats is not None:
            astats[f'{n}.q'].add(q); astats[f'{n}.k'].add(k); astats[f'{n}.v'].add(v)
        if aq is not None:
            q = fq_asym(q, *aq[f'{n}.q']); k = fq_asym(k, *aq[f'{n}.k']); v = fq_asym(v, *aq[f'{n}.v'])
        attn = (q * m.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        if astats is not None:
            astats[f'{n}.p'].add(attn)
        if aq is not None:
            attn = fq_asym(attn, *aq[f'{n}.p'])
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return m.proj(x)
    return fwd


def apply_variant(tr, variant, calib, skip_blocks=()):
    net = tr.network
    if variant == 'fp32':
        return
    rkey = 'mm' if variant == 'w8a8_mm' else 'pct'
    skip = lambda n: any(f'blocks.{b}.' in n for b in skip_blocks)
    for n, m in quant_modules(net):
        with torch.no_grad():
            m.weight.copy_(fq_weight(m.weight))
        if variant.startswith('w8a8') and not skip(n):
            lo, hi = calib['lin'][n][rkey]
            m.register_forward_pre_hook(lambda mod, inp, lo=lo, hi=hi: (fq_asym(inp[0], lo, hi),) + tuple(inp[1:]))
    if variant == 'w8a8_pct_attn':
        aq = {k: v['pct'] for k, v in calib['attn'].items()}
        for n, m in attn_modules(net):
            if skip(n):
                continue
            m.fused_attn = False; m._fq_name = n
            m.forward = make_attn_forward(m, None, aq)


# ---------------------------------------------------------------- evaluation
def worker(rank, a, seq_names):
    torch.cuda.set_device(rank % torch.cuda.device_count())
    torch.set_num_threads(1)
    tr = build_tracker(a.cfg)
    calib = torch.load(os.path.join(ROOT, a.calib)) if a.variant != 'fp32' else None
    apply_variant(tr, a.variant, calib, [int(b) for b in a.skip_blocks.split(',')] if a.skip_blocks else ())
    from lib.test.evaluation import get_dataset
    out = os.path.join(ROOT, a.out); os.makedirs(out, exist_ok=True)
    for seq in get_dataset(a.dataset):
        if seq.name not in seq_names:
            continue
        p = os.path.join(out, seq.name + '.txt')
        if os.path.exists(p):
            continue
        img = lambda f: cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        init = list(seq.ground_truth_rect[0])
        tr.initialize(img(seq.frames[0]), {'init_bbox': init})
        boxes = [init]
        with torch.no_grad():
            for f in seq.frames[1:]:
                boxes.append(tr.track(img(f))['target_bbox'])
        np.savetxt(p + '.tmp', np.array(boxes), delimiter='\t', fmt='%.2f')
        os.replace(p + '.tmp', p)


def do_run(a):
    import multiprocessing as mp
    sys.path.insert(0, REPO); os.chdir(os.path.join(REPO, 'tracking'))
    from lib.test.evaluation import get_dataset
    names = [s.name for s in get_dataset(a.dataset)]
    if a.seqs:
        names = [n for n in names if n in a.seqs.split(',')]
    chunks = [names[i::a.workers] for i in range(a.workers)]
    ctx = mp.get_context('spawn')
    ps = [ctx.Process(target=worker, args=(i, a, set(c))) for i, c in enumerate(chunks)]
    for p in ps: p.start()
    for p in ps: p.join()
    print('done', a.variant, a.dataset, len(names))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['calib', 'run'])
    p.add_argument('--cfg', default='deit_tiny_distilled_patch16_224')
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--calib', default='analysis/fq_calib_D.pt')
    p.add_argument('--variant', default='fp32')
    p.add_argument('--dataset', default='uav123')
    p.add_argument('--out', default='analysis/fq_tmp')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seqs', default='')
    p.add_argument('--skip-blocks', dest='skip_blocks', default='')
    a = p.parse_args()
    os.environ.pop('SOT_ADAPT_SF', None); os.environ.pop('SOT_MOTION', None)
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    if a.mode == 'calib':
        do_calib(a)
    else:
        do_run(a)

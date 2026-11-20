"""실험(2026-11-13): 학습 없는 상태 흐름 추적 (공식 ORTrack 가중치 그대로).
TRACK → (존재 의심) → 전역 재탐색 → 못 찾으면 LOST(KF 예측 유지, 범위 확대, 주기 재탐색) → 후보가 엄격 기준을 연속 통과하면 TRACK 복귀.
존재 판단: 점수 지도 최댓값 s1, 두 번째 봉우리 비 r2, 봉우리 집중도 sharp, 예측 위치 토큰-템플릿 유사도 sim, sim/최근평균 drop.
env SOT_SM="s1:sim:drop:r2" (존재 기준, 기본 0.30:0.50:0.80:0.70), SOT_SM_STRICT=복귀 시 기준 배수(기본 1.2),
SOT_SM_CONFIRM=연속 확인 프레임(2), SOT_SM_EVERY=LOST 재탐색 간격(5), SOT_SM_FULL=전체 타일 훑기 간격(30), SOT_SM_LOG=dir.
KF 는 SOT_MOTION=kf (CMCKalman) 필요."""
import math, os
import numpy as np
import torch
import torch.nn.functional as F
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box


def _post(trk, out, b, state, rf):
    """배치 출력의 b 번째에서 후보(최대 3개)와 특징 추출. 선택 규칙: 창 적용 응답 최댓값 → 3x3 이웃 제거 → 반복(응답<=1e-4 중단)."""
    with torch.no_grad():
        sm = out['score_map'][b, 0]; Sz = sm.shape[-1]
        resp = (trk.output_window[0, 0] * sm) if trk.output_window.dim() == 4 else trk.output_window * sm
        feat = out['backbone_feat']; feat = feat[-1] if isinstance(feat, list) else feat
        nz = trk.network.feat_len_t if hasattr(trk.network, 'feat_len_t') else 64
        zt = F.normalize(feat[b, :nz].mean(0), dim=0)
        xt = F.normalize(feat[b, nz:nz + Sz * Sz], dim=1)
        flat = resp.flatten().clone()
        ar = torch.arange(Sz * Sz, device=flat.device); ay, ax = ar // Sz, ar % Sz
        idx, val = [], []
        for _ in range(3):
            v, jj = flat.max(0); idx.append(jj); val.append(v)
            near = ((ay - jj // Sz).abs() <= 1) & ((ax - jj % Sz).abs() <= 1)
            flat = torch.where(near, torch.zeros_like(flat), flat)
        idx = torch.stack(idx); val = torch.stack(val)
        y0, x0 = idx // Sz, idx % Sz
        smp = F.pad(sm[None, None], (1, 1, 1, 1))[0, 0]
        nbs = torch.stack([smp[y0 + dy, x0 + dx] for dy in range(3) for dx in range(3)]).sum(0)
        size = out['size_map'][b].flatten(1)[:, idx]; off = out['offset_map'][b].flatten(1)[:, idx]
        bb = torch.stack([(x0.float() + off[0]) / Sz, (y0.float() + off[1]) / Sz, size[0], size[1]], 1) * trk.params.search_size / rf
        sims = xt[idx] @ zt
        pack = torch.cat([val[:, None], sm.flatten()[idx][:, None], nbs[:, None] / (sm.sum() + 1e-6), sims[:, None], bb], 1).cpu().numpy()
        fs = xt[idx].cpu().numpy()
    cx, cy = state[0] + state[2] / 2, state[1] + state[3] / 2
    h0 = 0.5 * trk.params.search_size / rf
    cands = []
    for k in range(3):
        if pack[k, 0] <= 1e-4: break
        q = pack[k, 4:8]
        cands.append(dict(box=[q[0] + cx - h0 - q[2] / 2, q[1] + cy - h0 - q[3] / 2, q[2], q[3]], s=float(pack[k, 1]),
                          sharp=float(pack[k, 2]), sim=float(pack[k, 3]), f=fs[k]))
    r2 = cands[1]['s'] / max(cands[0]['s'], 1e-6) if len(cands) > 1 else 0.0
    for c in cands: c['r2'] = r2
    return cands, (cx, cy, 2 * h0)


def _local_many(trk, image, states, sf):
    """여러 위치(같은 배율)를 한 번의 신경망 호출(batch)로 처리."""
    xs, ms, rfs = [], [], []
    for st in states:
        x_arr, rf, x_m = sample_target(image, st, sf, output_sz=trk.params.search_size)
        sp = trk.preprocessor.process(x_arr, x_m); xs.append(sp.tensors); rfs.append(rf)
    with torch.no_grad():
        X = torch.cat(xs, 0); Z = trk.z_dict1.tensors.expand(X.shape[0], -1, -1, -1)
        out = trk.network.forward(template=Z, search=X)
    return [_post(trk, out, b, st, rf) for b, (st, rf) in enumerate(zip(states, rfs))]


def _local(trk, image, state, sf):
    """state 중심 crop(배율 sf) 1회 추론 → 후보(최대 3개)와 특징."""
    return _local_many(trk, image, [state], sf)[0]


def _zfail(S, c, drop):
    """존재 판단 모델(SOT_SM_MODEL json, 로지스틱): 실패 점수 z (클수록 표적 아님)."""
    m = S['model']; x = np.array([c['s'], c['sim'], c['sharp'], c['r2'], drop])
    return float(((x - m['mu']) / m['sd']) @ m['w'] + m['b'])


def _ok(c, th, drop, k=1.0):
    s1, sim, dr, _ = th
    return c['s'] >= s1 * k and c['sim'] >= min(sim * k, 0.99) and drop >= dr  # 복귀 엄격 배수는 s1·sim 에만


def track_sm(trk, image):
    """중재(arbitration) 방식: ORTrack 국소 박스가 기본 출력. 의심(존재 모델 z>thr)일 때만 넓은 재탐색 후보를 만들고,
    후보가 z 기준 margin 이상 확실히 나을 때만 교체. 의심이 이어지면 재탐색 범위 확대(2→4배→화면 타일). KF 는 확신 프레임에서만 갱신."""
    H, W, _ = image.shape
    E = os.environ
    trk.frame_id += 1
    if not hasattr(trk, 'sm'):
        m = __import__('json').load(open(E['SOT_SM_MODEL']))
        trk.sm = dict(model={k: np.array(v) if isinstance(v, list) else v for k, v in m.items()},
                      margin=float(E.get('SOT_SM_MARGIN', '1.0')), every=int(E.get('SOT_SM_EVERY', '1')),
                      full=int(E.get('SOT_SM_FULL', '15')), lam=float(E.get('SOT_KF_LAMBDA', '9')),
                      r2=float(E.get('SOT_SM_R2', '0.7')), log=[] if E.get('SOT_SM_LOG') else None)
    S = trk.sm
    if trk.frame_id == 1:
        S.update(susp=0, sim_ema=None, oov=False, vstate=list(trk.state), oov_n=0, oov_pend=0, T=[], D=[])
        fl = E.get('SOT_FLOOR', '10')  # 'auto' = min(10, 첫 프레임 표적 짧은 변)
        S['floor'] = int(min(10, max(1, math.floor(min(trk.state[2], trk.state[3]))))) if fl == 'auto' else int(fl)  # 제한이 실제 표적보다 크지 않게
        if S['log'] is not None: S['log'] = []
    kf = trk.motion
    oov_on = bool(E.get('SOT_SM_OOV'))
    base = S['vstate'] if (oov_on and S['oov']) else trk.state   # 화면 밖일 때는 잘리지 않은 가상 위치로 CMC 예측
    pred = kf.predict(image, base) if kf is not None else base
    if oov_on:
        pc = (pred[0] + pred[2] / 2, pred[1] + pred[3] / 2)
        outside = pc[0] < 0 or pc[0] >= W or pc[1] < 0 or pc[1] >= H
        if S['oov'] or outside:
            S['vstate'] = list(pred)
            if not S['oov']:
                S['oov'], S['oov_n'], S['oov_pend'] = True, 0, 0
            S['oov_n'] += 1
            # 가장자리에서 재진입 확인: 가상 위치를 화면 안으로 당긴 지점에서 국소 탐색
            hw = math.sqrt(pred[2] * pred[3]) * sf / 2
            ex = min(max(pc[0], hw * 0.5), W - hw * 0.5); ey = min(max(pc[1], hw * 0.5), H - hw * 0.5)
            probe = [ex - pred[2] / 2, ey - pred[3] / 2, pred[2], pred[3]]
            cs, _ = _local(trk, image, probe, sf)
            cc = cs[0] if cs else None
            ok = cc is not None and _zfail(S, cc, cc['sim'] / S['sim_ema'] if S['sim_ema'] else 1.0) < S['model']['thr'] - 1.0
            S['oov_pend'] = S['oov_pend'] + 1 if ok else 0
            if S['oov_pend'] >= 2 and not outside:   # 2프레임 연속 확인 + 예측도 화면 안 → 복귀
                S['oov'] = False; trk.state = clip_box(cc['box'], H, W, margin=S['floor'])
                if kf is not None: kf.update(trk.state)
            else:
                trk.state = clip_box(probe, H, W, margin=S['floor'])
                if kf is not None and hasattr(kf, 'prev_gray'): pass
            if S['log'] is not None:
                S['log'].append([trk.frame_id, 5 if S['oov'] else 3, cc['s'] if cc else 0, cc['sim'] if cc else 0, 0, 0, 1, 0] + [0.0] * 10)
            return {"target_bbox": trk.state, "oov": S['oov']}
    sf = trk.params.search_factor
    thr = S['model']['thr']

    def z(c):
        dr = c['sim'] / S['sim_ema'] if S['sim_ema'] else 1.0
        return _zfail(S, c, dr)

    def pick(cands):
        if len(cands) > 1 and cands[0]['r2'] >= S['r2'] and kf is not None and hasattr(kf, 'd2'):
            return max(cands, key=lambda c: c['s'] * math.exp(-0.5 * min(kf.d2(c['box']), 50.0) / S['lam']))
        return cands[0]

    tw = [0.0, -1, 0, 0, 0, 0, 0, 0, 0, 0]  # 두 위치 기록: [카메라 이동/crop, 선택(-1 단일/0 CMC/1 직전), CMC후보 box4, 직전후보 box4]
    two = E.get('SOT_SM_TWO')  # "두 위치 비교": 직전 위치 vs CMC 이동 위치, 떨어져 있으면 둘 다 보고 z 낮은 쪽 (값 = crop 한 변 대비 최소 거리)
    if two and kf is not None:
        p0 = trk.state
        d = math.hypot(pred[0] + pred[2] / 2 - (p0[0] + p0[2] / 2), pred[1] + pred[3] / 2 - (p0[1] + p0[3] / 2))
        side = math.sqrt(p0[2] * p0[3]) * sf
        tw[0] = d / max(side, 1e-6)
        if d > float(two) * side:  # 두 위치를 배치 2로 한 번에
            (cands, _), (cands0, _) = _local_many(trk, image, [pred, p0], sf); ncall = 2
            c = pick(cands); zc = z(c); c0 = pick(cands0); z0 = z(c0); tw[1] = 0; tw[2:6] = c['box']; tw[6:10] = c0['box']
            if z0 < zc: c, zc = c0, z0; tw[1] = 1
        else:
            cands, _ = _local(trk, image, pred, sf); ncall = 1
            c = pick(cands); zc = z(c)
    else:
        cands, _ = _local(trk, image, pred, sf); ncall = 1
        c = pick(cands); zc = z(c)
    status = 0
    if zc > thr:
        S['susp'] += 1
        alts = []
        if S['susp'] % S['every'] == 0:
            scale = 2.0 if S['susp'] < 10 else 4.0
            cs, _ = _local(trk, image, pred, sf * scale); ncall += 1
            if cs: alts.append(pick(cs))
            if S['susp'] >= 10 and S['susp'] % S['full'] == 0:  # 의심이 길면 화면 2x2 타일
                side = max(H, W) / 2 / sf
                for ty in (0.25, 0.75):
                    for tx in (0.25, 0.75):
                        cs, _ = _local(trk, image, [tx * W - side / 2, ty * H - side / 2, side, side], sf); ncall += 1
                        if cs: alts.append(pick(cs))
        if alts:
            a = min(alts, key=z); za = z(a)
            veto = False
            if E.get('SOT_SM_DVETO') and S['D'] and a.get('f') is not None:  # 방해물 기억과 더 닮은 대안은 교체 거절
                fa = a['f'] / (np.linalg.norm(a['f']) + 1e-6)
                veto = max(float(fa @ v) for v in S['D']) > max([float(fa @ v) for v in S['T']], default=0.0)
            if za < thr and za < zc - S['margin'] and not veto:
                c, zc, status = a, za, 1   # 교체 (재획득)
        if status == 0: status = 4        # 의심이지만 국소 박스 유지
    else:
        S['susp'] = 0
    trk.state = clip_box(c['box'], H, W, margin=S['floor']); S['vstate'] = list(trk.state)
    if zc <= thr:
        if E.get('SOT_SM_DVETO') and c.get('f') is not None:  # 메모리: 확신 프레임의 표적 특징, 두 번째 봉우리(닮은 물체) 특징
            f = c['f'] / (np.linalg.norm(c['f']) + 1e-6)
            if not S['T'] or trk.frame_id % 5 == 0: S['T'] = (S['T'] + [f])[-5:]
            if len(cands) > 1 and cands[0]['r2'] >= 0.6 and cands[1].get('f') is not None:
                g = cands[1]['f'] / (np.linalg.norm(cands[1]['f']) + 1e-6); S['D'] = (S['D'] + [g])[-5:]
        if kf is not None: kf.update(trk.state)
        S['sim_ema'] = c['sim'] if S['sim_ema'] is None else 0.9 * S['sim_ema'] + 0.1 * c['sim']
    elif kf is not None and hasattr(kf, 'cmc'):
        kf.cmc.update(trk.state)
    if S['log'] is not None:
        S['log'].append([trk.frame_id, status, c['s'], c['sim'], c['sharp'], c['r2'], ncall, zc] + tw)
    return {"target_bbox": trk.state}


def track_beam(trk, image):
    """가설 트리(지연 결정) + 방해물 메모리. env SOT_SM_BEAM=K(가설 수, 기본 2).
    - 모든 가설 위치를 같은 CMC 변환 H 로 옮김(카메라 움직임은 공통). 크게 움직이면 각 가설을 '따라감(제자리)/안 따라감(H 이동)'으로 분기.
    - 의심 시 넓은 재탐색 후보를 새 가설로 추가(즉시 교체 없음).
    - 각 가설 점수 = 누적(지수평균) [ -z(존재) + a·(표적 메모리 유사도 - 방해물 메모리 유사도) ]. 1위 출력, 1위와 격차 큰 가설은 제거.
    - 메모리: 표적 = 첫 템플릿 + 확신 프레임 특징(최대 5), 방해물 = 확신 프레임의 두 번째 봉우리 특징(r2 높을 때, 최대 5)."""
    H, W, _ = image.shape
    E = os.environ
    trk.frame_id += 1
    if not hasattr(trk, 'sm'):
        m = __import__('json').load(open(E['SOT_SM_MODEL']))
        trk.sm = dict(model={k: np.array(v) if isinstance(v, list) else v for k, v in m.items()},
                      K=int(E.get('SOT_SM_BEAM', '2')), two=float(E.get('SOT_SM_TWO', '0.05')), a=float(E.get('SOT_SM_APP', '2.0')),
                      gap=float(E.get('SOT_SM_GAP', '1.5')), mom=float(E.get('SOT_SM_MOM', '0.6')),
                      log=[] if E.get('SOT_SM_LOG') else None)
    S = trk.sm; thr = S['model']['thr']; sf = trk.params.search_factor; kf = trk.motion
    floor = int(E.get('SOT_FLOOR', '10'))
    if trk.frame_id == 1:
        S.update(hyps=[dict(box=list(trk.state), acc=0.0)], T=[], D=[], sim_ema=None)
        if S['log'] is not None: S['log'] = []
    # CMC 변환 (한 번만): 1위 가설 기준으로 계산
    best0 = S['hyps'][0]['box']
    pb = kf.predict(image, best0) if kf is not None else best0
    Hm = getattr(kf, 'H', None) if kf is not None else None

    def warp(b):
        if Hm is None: return list(b)
        c = np.array([b[0] + b[2] / 2, b[1] + b[3] / 2, 1.0]); p = Hm @ c; p = p[:2] / p[2]
        return [p[0] - b[2] / 2, p[1] - b[3] / 2, b[2], b[3]]

    def app(f):
        if f is None: return 0.0
        fn = f / (np.linalg.norm(f) + 1e-6)
        t = max([float(fn @ v) for v in S['T']], default=0.0); d = max([float(fn @ v) for v in S['D']], default=0.0)
        return t - d if S['D'] else t - 0.5

    def zc(c):
        dr = c['sim'] / S['sim_ema'] if S['sim_ema'] else 1.0
        return _zfail(S, c, dr)

    # 1) 위치 후보 생성
    pos = []
    for h in S['hyps']:
        wb = warp(h['box'])
        pos.append((wb, h['acc']))
        d = math.hypot(wb[0] - h['box'][0], wb[1] - h['box'][1]); side = math.sqrt(h['box'][2] * h['box'][3]) * sf
        if d > S['two'] * side:
            pos.append((list(h['box']), h['acc']))   # 카메라가 따라갔다는 가설
    # 2) 각 위치에서 ORTrack
    new, ncall, cand2 = [], 0, None
    for b, acc in pos:
        cs, _ = _local(trk, image, clip_box(b, H, W, margin=floor), sf); ncall += 1
        if not cs: continue
        c = cs[0]; z = zc(c); step = -z + S['a'] * app(c.get('f'))
        new.append(dict(box=c['box'], acc=S['mom'] * acc + (1 - S['mom']) * step, z=z, c=c, cs=cs))
    new.sort(key=lambda h: -h['acc'])
    # 3) 의심이면 넓은 재탐색 후보를 새 가설로 추가
    if new and new[0]['z'] > thr:
        cs, _ = _local(trk, image, clip_box(new[0]['box'], H, W, margin=floor), sf * 3.0); ncall += 1
        if cs:
            c = cs[0]; z = zc(c)
            if z < thr:
                step = -z + S['a'] * app(c.get('f'))
                new.append(dict(box=c['box'], acc=new[0]['acc'] - 0.5 + (1 - S['mom']) * step, z=z, c=c, cs=cs))
                new.sort(key=lambda h: -h['acc'])
    if not new:
        new = [dict(box=list(best0), acc=-9, z=9, c=None, cs=[])]
    # 4) 중복 가설 합치기 + 격차 큰 가설 제거 + 상위 K
    kept = []
    for h in new:
        ctr = (h['box'][0] + h['box'][2] / 2, h['box'][1] + h['box'][3] / 2); r = math.sqrt(max(h['box'][2] * h['box'][3], 1.0))
        if any(math.hypot(ctr[0] - (k['box'][0] + k['box'][2] / 2), ctr[1] - (k['box'][1] + k['box'][3] / 2)) < r for k in kept): continue
        if kept and h['acc'] < kept[0]['acc'] - S['gap']: continue
        kept.append(h)
        if len(kept) >= S['K']: break
    best = kept[0]
    trk.state = clip_box(best['box'], H, W, margin=floor)
    S['hyps'] = [dict(box=clip_box(h['box'], H, W, margin=floor), acc=h['acc']) for h in kept]
    # 5) 메모리 갱신 (확신 프레임만)
    c = best['c']
    if c is not None and best['z'] < thr:
        S['sim_ema'] = c['sim'] if S['sim_ema'] is None else 0.9 * S['sim_ema'] + 0.1 * c['sim']
        f = c['f'] / (np.linalg.norm(c['f']) + 1e-6)
        if not S['T'] or trk.frame_id % 5 == 0:
            S['T'] = (S['T'] + [f])[-5:]
        cs = best['cs']
        if len(cs) > 1 and cs[0]['r2'] >= 0.6:
            g = cs[1]['f'] / (np.linalg.norm(cs[1]['f']) + 1e-6); S['D'] = (S['D'] + [g])[-5:]
        if kf is not None: kf.update(trk.state)
    if S['log'] is not None:
        S['log'].append([trk.frame_id, len(kept), best['z'], best['acc'], ncall, len(S['T']), len(S['D'])])
    return {"target_bbox": trk.state}


def track_simple(trk, image):
    """핵심판(2026-11-13): 직전 위치와 CMC 이동 위치를 배치 2로 함께 추적, 점수 높은 쪽 채택. 최소 박스 = min(10, 표적 짧은 변).
    학습·임계값 없음. env SOT_SIMPLE=1, SOT_MOTION=cmc|phase."""
    H, W, _ = image.shape
    trk.frame_id += 1
    if trk.frame_id == 1:
        fl = os.environ.get('SOT_FLOOR', 'auto')
        trk.sm_floor = int(min(10, max(1, math.floor(min(trk.state[2], trk.state[3]))))) if fl == 'auto' else int(fl)
        trk.sm_log = [] if os.environ.get('SOT_SM_LOG') else None
        trk.sm_alpha = 0.5   # 피드백: 이 영상에서 CMC 위치가 맞아 온 정도(누적)
    kf = trk.motion; sf = trk.params.search_factor
    p0 = list(trk.state)
    pred = kf.predict(image, p0) if kf is not None else p0
    mode = os.environ.get('SOT_SIMPLE_MODE', 'AB')  # Ablation: AB(제안) | B(CMC 위치만) | A(직전 위치만 = ORTrack)
    states = {'AB': [pred, p0], 'B': [pred], 'A': [p0]}[mode] if kf is not None else [p0]
    # 화면 밖 추적(env SOT_OOV=1): 표적 중심을 화면 밖까지 H 로 계속 옮긴 가상 위치 v.
    # v 가 화면 밖(박스가 완전히 벗어남)이면 추적을 멈추고(방해물로 옮겨 가지 않음), v 가 다시 들어오면 그 자리에서 찾는다.
    if os.environ.get('SOT_OOV') and kf is not None and mode == 'AB':
        if trk.frame_id == 1 or not hasattr(trk, 'oov_v'):
            trk.oov_v, trk.oov = np.array([p0[0] + p0[2] / 2, p0[1] + p0[3] / 2]), False
        Hm = getattr(kf, 'H', None); v = trk.oov_v
        if Hm is not None:
            q = Hm @ np.array([v[0], v[1], 1.0]); v = q[:2] / q[2]
        trk.oov_v = v
        mx, my = p0[2] / 2 * float(os.environ.get('SOT_OOV_M', '1')), p0[3] / 2 * float(os.environ.get('SOT_OOV_M', '1'))
        if v[0] < -mx or v[1] < -my or v[0] > W - 1 + mx or v[1] > H - 1 + my:   # 화면 밖
            if not trk.oov:
                trk.oov_last = list(p0)   # 마지막 추적 위치(짐벌이 따라갔다면 표적은 여기 남아 있음)
            trk.oov = True
            g = float(os.environ.get('SOT_OOV_G', '0'))   # >0: 마지막 위치 점수가 평소 점수의 g 배 이상이면 화면 밖 판단 취소
            if g > 0:
                rr = _local_many(trk, image, [trk.oov_last], sf)[0]
                if rr[0] and rr[0][0]['s'] >= g * getattr(trk, 's_ema', 1.0):
                    trk.oov = False
                    trk.state = clip_box(rr[0][0]['box'], H, W, margin=trk.sm_floor)
                    trk.oov_v = np.array([trk.state[0] + trk.state[2] / 2, trk.state[1] + trk.state[3] / 2])
                    if os.environ.get('SOT_GATE_LOG'):
                        if trk.frame_id == 1: trk.sm = {'log': []}
                        trk.sm['log'].append([trk.frame_id, -1] + ([np.nan] * 10 if os.environ.get('SOT_SEL_LOG') else []))
                    return {"target_bbox": trk.state}
            cx, cy = min(max(v[0], 0.0), W - 1.0), min(max(v[1], 0.0), H - 1.0)
            trk.state = clip_box([cx - p0[2] / 2, cy - p0[3] / 2, p0[2], p0[3]], H, W, margin=trk.sm_floor)
            if os.environ.get('SOT_GATE_LOG'):
                if trk.frame_id == 1: trk.sm = {'log': []}
                trk.sm['log'].append([trk.frame_id, 0] + ([np.nan] * 10 if os.environ.get('SOT_SEL_LOG') else []))
            return {"target_bbox": trk.state}
        if trk.oov:   # 다시 들어옴: 가상 위치에서 검색(직전 위치 대신)
            trk.oov = False
            states = [[v[0] - p0[2] / 2, v[1] - p0[3] / 2, p0[2], p0[3]], pred]
    gate = float(os.environ.get('SOT_GATE', '0'))  # 조건부 B: CMC 이동이 crop 한 변의 gate 배 미만이면 B 생략 (0 = 항상 B)
    if gate > 0 and mode == 'AB' and len(states) == 2:
        d = math.hypot(pred[0] + pred[2] / 2 - p0[0] - p0[2] / 2, pred[1] + pred[3] / 2 - p0[1] - p0[3] / 2)
        if d < gate * sf * math.sqrt(p0[2] * p0[3]):
            states = [p0]
    # 기억 가설 C(env SOT_ANCHOR=감쇠율, 0=끔): 마지막으로 확실했던 위치를 카메라 움직임(H)만큼 계속 옮겨 둠.
    # 추적이 방해물로 옮겨 가도 C 는 원래 표적 자리를 가리키므로, 표적이 다시 보이면 점수 경쟁으로 되돌아온다.
    dec = float(os.environ.get('SOT_ANCHOR', '0'))
    ic = -1
    if dec > 0 and kf is not None:
        if trk.frame_id == 1 or not hasattr(trk, 'anc'):
            trk.anc, trk.anc_s, trk.anc_s0 = list(p0), 1.0, 1.0
        else:
            Hm = getattr(kf, 'H', None); a = trk.anc
            cx, cy = a[0] + a[2] / 2, a[1] + a[3] / 2
            if Hm is not None:
                q = Hm @ np.array([cx, cy, 1.0]); cx, cy = q[0] / q[2], q[1] / q[2]
            cx, cy = min(max(cx, 0.0), W - 1.0), min(max(cy, 0.0), H - 1.0)
            trk.anc = [cx - a[2] / 2, cy - a[3] / 2, a[2], a[3]]
            r = 0.25 * sf * math.sqrt(p0[2] * p0[3])   # A·B 와 crop 한 변의 1/4 이상 떨어질 때만 따로 검색
            if all(math.hypot(cx - st[0] - st[2] / 2, cy - st[1] - st[3] / 2) > r for st in states):
                states = states + [trk.anc]; ic = len(states) - 1
    res = _local_many(trk, image, states, sf)
    nh = len(states)
    gs = float(os.environ.get('SOT_GATE_S', '0'))  # B를 생략했어도 A 점수가 gs 미만이면 B를 이어서 검색
    if gs > 0 and nh == 1 and mode == 'AB' and kf is not None and (not res[0][0] or res[0][0][0]['s'] < gs):
        res = res + _local_many(trk, image, [pred], sf); nh = 2
    fb = float(os.environ.get('SOT_FB', '0'))  # 피드백 세기 β (0 = 끔)
    if fb > 0 and len(res) == 2 and res[0][0] and res[1][0]:
        a = trk.sm_alpha
        sc = [res[0][0][0]['s'] + fb * (a - 0.5), res[1][0][0]['s'] + fb * (0.5 - a)]   # 0: CMC 위치, 1: 직전 위치
        k = 0 if sc[0] >= sc[1] else 1
        best = res[k][0][0]
        if abs(res[0][0][0]['s'] - res[1][0][0]['s']) > 0.05:   # 둘이 실제로 다를 때만 기억 갱신
            trk.sm_alpha = 0.9 * a + 0.1 * (1.0 if res[0][0][0]['s'] > res[1][0][0]['s'] else 0.0)
    else:
        best = max((r[0][0] for k, r in enumerate(res) if r[0] and k != ic), key=lambda c: c['s'], default=None)
        if ic >= 0 and res[ic][0]:   # 기억 가설 C 는 기억할 당시 점수의 g 배 이상일 때만 경쟁 (env SOT_ANCHOR_G, 0 = 조건 없음)
            cC = res[ic][0][0]; g = float(os.environ.get('SOT_ANCHOR_G', '0'))
            lost_ok = (not os.environ.get('SOT_ANCHOR_LOST')) or best is None or best['s'] < g * trk.anc_s0   # LOST=1: A·B 가 약할 때만
            if lost_ok and cC['s'] >= g * trk.anc_s0 and (best is None or cC['s'] > best['s']):
                best = cC
    if best is not None:
        trk.state = clip_box(best['box'], H, W, margin=trk.sm_floor)
        if dec > 0 and kf is not None:   # 기억 갱신: 현재 점수가 감쇠된 기억 점수 이상이면 현재 위치로 교체
            trk.anc_s *= dec
            if best['s'] >= trk.anc_s:
                trk.anc, trk.anc_s, trk.anc_s0 = list(trk.state), best['s'], best['s']
    if os.environ.get('SOT_OOV') and hasattr(trk, 'oov_v'):   # 추적 중에는 가상 위치 = 추적 중심, 평소 점수 EMA
        trk.oov_v = np.array([trk.state[0] + trk.state[2] / 2, trk.state[1] + trk.state[3] / 2])
        if best is not None:
            trk.s_ema = best['s'] if not hasattr(trk, 's_ema') else 0.9 * trk.s_ema + 0.1 * best['s']
    if os.environ.get('SOT_GATE_LOG'):  # 조건부 B 실행 기록: [frame, 검색 가설 수]
        if trk.frame_id == 1: trk.sm = {'log': []}
        row = [trk.frame_id, nh]
        if os.environ.get('SOT_SEL_LOG'):   # 가설별 [중심x, 중심y, 점수] ×3 + 고른 번호 (분석용)
            for k in range(3):
                if k < len(res):
                    st = states[k] if k < len(states) else pred
                    row += [st[0] + st[2] / 2, st[1] + st[3] / 2, res[k][0][0]['s'] if res[k][0] else -1]
                else:
                    row += [np.nan, np.nan, np.nan]
            row += [next((k for k, r in enumerate(res) if r[0] and best is not None and r[0][0] is best), -1)]
        trk.sm['log'].append(row)
    if trk.sm_log is not None:
        trk.sm_log.append([trk.frame_id] + [r[0][0]['s'] if r[0] else 0 for r in res])
    return {"target_bbox": trk.state}


def track_reacq(trk, image):
    """핵심판 + DAM4SAM 식 다시 찾기(2026-11-13). env SOT_REACQ=1.
    추적: 두 위치(CMC 이동 / 직전)를 배치 2 로, 점수 + 피드백 α 로 선택. 신원 = 표적 기억 유사도 - 방해물 기억 유사도 < 0 이면 거절.
    놓침(점수<τ 또는 신원 거절): 박스는 CMC 로만 이동, 기억 갱신 중단. crop 을 놓친 시간에 따라 확대, 10프레임마다 화면 2x2 훑기(배치 4).
    복귀: 점수>=τ_re 이고 신원 통과, 같은 자리에서 2프레임 연속. env SOT_TAU(기본 0.5), SOT_TAU_RE(0.6), SOT_FB(0.2), SOT_FLOOR(10)."""
    H, W, _ = image.shape
    E = os.environ
    trk.frame_id += 1
    sf = trk.params.search_factor; kf = trk.motion
    if trk.frame_id == 1:
        trk.rq = dict(alpha=0.5, lost=0, pend=None, pend_n=0, T=[], D=[], floor=int(E.get('SOT_FLOOR', '10')),
                      tau=float(E.get('SOT_TAU', '0.5')), tau_re=float(E.get('SOT_TAU_RE', '0.6')), fb=float(E.get('SOT_FB', '0.2')),
                      log=[] if E.get('SOT_SM_LOG') else None)
    Q = trk.rq
    nrm = lambda f: f / (np.linalg.norm(f) + 1e-6)

    def ident(c):
        f = nrm(c['f'])
        if not Q['T']: return 1.0
        t = max(float(f @ v) for v in Q['T']); d = max([float(f @ v) for v in Q['D']], default=-1.0)
        return t - d if Q['D'] else t

    p0 = list(trk.state)
    pred = kf.predict(image, p0) if kf is not None else p0
    ncall = 0; status = 0
    if Q['lost'] == 0:
        res = _local_many(trk, image, [pred, p0], sf); ncall = 1
        cs = [r[0][0] if r[0] else None for r in res]
        ok = [c is not None and c['s'] >= Q['tau'] and ident(c) >= 0 for c in cs]
        if any(ok):
            sc = [(c['s'] + Q['fb'] * ((Q['alpha'] - 0.5) if k == 0 else (0.5 - Q['alpha']))) if ok[k] else -9 for k, c in enumerate(cs)]
            k = int(np.argmax(sc)); c = cs[k]
            if all(ok) and abs(cs[0]['s'] - cs[1]['s']) > 0.05:
                Q['alpha'] = 0.9 * Q['alpha'] + 0.1 * (1.0 if cs[0]['s'] > cs[1]['s'] else 0.0)
            trk.state = clip_box(c['box'], H, W, margin=Q['floor'])
            # 기억 갱신 (확실한 프레임만)
            if not Q['T'] or trk.frame_id % 5 == 0: Q['T'] = (Q['T'] + [nrm(c['f'])])[-5:]
            alts = res[k][0]
            if len(alts) > 1 and alts[0]['r2'] >= 0.6: Q['D'] = (Q['D'] + [nrm(alts[1]['f'])])[-5:]
        else:
            Q['lost'] = 1; status = 2
            trk.state = clip_box(pred, H, W, margin=Q['floor'])
    else:
        Q['lost'] += 1; status = 2
        trk.state = clip_box(pred, H, W, margin=Q['floor'])
        scale = min(1.0 + Q['lost'] / 10.0, 2.5)
        states = [trk.state]
        if Q['lost'] % 10 == 0:   # 화면 2x2 훑기
            side = max(H, W) / 2 / sf
            states += [[tx * W - side / 2, ty * H - side / 2, side, side] for ty in (0.25, 0.75) for tx in (0.25, 0.75)]
        res = _local_many(trk, image, states[:1], sf * scale); ncall = 1
        if len(states) > 1:
            res += _local_many(trk, image, states[1:], sf); ncall += 1
        cs = [r[0][0] for r in res if r[0]]
        cs = [c for c in cs if c['s'] >= Q['tau_re'] and ident(c) >= 0]
        if cs:
            c = max(cs, key=lambda c: c['s']); ctr = (c['box'][0] + c['box'][2] / 2, c['box'][1] + c['box'][3] / 2)
            r = 2 * max(math.sqrt(c['box'][2] * c['box'][3]), 8)
            Q['pend_n'] = Q['pend_n'] + 1 if Q['pend'] is not None and math.hypot(ctr[0] - Q['pend'][0], ctr[1] - Q['pend'][1]) < r else 1
            Q['pend'] = ctr
            if Q['pend_n'] >= 2:
                trk.state = clip_box(c['box'], H, W, margin=Q['floor']); Q['lost'] = 0; Q['pend'] = None; Q['pend_n'] = 0; status = 3
            elif Q['pend_n'] == 1:
                trk.state = clip_box([ctr[0] - p0[2] / 2, ctr[1] - p0[3] / 2, p0[2], p0[3]], H, W, margin=Q['floor'])  # 다음 프레임에 그 자리 확인
        else:
            Q['pend'] = None; Q['pend_n'] = 0
    if Q['log'] is not None:
        Q['log'].append([trk.frame_id, status, Q['lost'], Q['alpha'], ncall, len(Q['T']), len(Q['D'])])
    return {"target_bbox": trk.state}

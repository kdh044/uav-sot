"""논문용 그림 2장 (PNG, 300 dpi).

fig_failure_cause.png : 실패 원인 (검색영역 안 혼동 / 밖 이탈) 누적 막대 — 수치 직접 표기
fig_speed_acc.png     : BioDrone val AUC vs GPU FPS (원본 vs +CMC, 우리)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'figs', 'paper'); os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'figure.facecolor': SURF, 'axes.facecolor': SURF})


def ev(tag):
    e = list(np.load(os.path.join(ROOT, 'analysis', f'failure_{tag}.npy'), allow_pickle=True))
    return sum(x['inside'] for x in e), sum(not x['inside'] for x in e)


# --- 실패 원인
groups = [('ORTrack\nStandard-4', 'ORTrack-DeiT'), ('ORTrack\nBioDrone', 'ORTrack-DeiT-bio'),
          ('Ours\nBioDrone', 'tiny-bio')]
ins, outs = zip(*[ev(t) for _, t in groups])
fig, ax = plt.subplots(figsize=(3.4, 2.8))
x = np.arange(len(groups)); w = 0.55
ax.bar(x, ins, w, color=BLUE, label='Distractor in search region', edgecolor=SURF, linewidth=2)
ax.bar(x, outs, w, bottom=ins, color=ORANGE, label='Target out of search region', edgecolor=SURF, linewidth=2)
for i in range(len(x)):
    ax.text(x[i], ins[i] / 2, str(ins[i]), ha='center', va='center', color='white', fontsize=8)
    if outs[i] >= 25:
        ax.text(x[i], ins[i] + outs[i] / 2, str(outs[i]), ha='center', va='center', color='white', fontsize=8)
    top = str(ins[i] + outs[i]) if outs[i] >= 25 else f'{ins[i] + outs[i]} (out {outs[i]})'
    ax.text(x[i], ins[i] + outs[i] + 6, top, ha='center', va='bottom', color=INK, fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([g for g, _ in groups])
ax.set_ylabel('Tracking failures')
ax.spines[['top', 'right']].set_visible(False); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_ylim(0, max(np.add(ins, outs)) * 1.12)
ax.legend(frameon=False, fontsize=7, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=1)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_failure_cause.png'), dpi=300); plt.close(fig)

# --- 속도-정확도 (BioDrone val AUC, 보이는 프레임; FPS = 서버 GPU, crop·전처리 포함)
CMC_MS = 8.04 - 2.97  # ORTrack 에서 측정한 CMC(640px) 추가 지연
models = [  # name, base latency ms, base AUC, +CMC AUC
    ('ORTrack', 3.25, 26.6, 31.3), ('ORTrack-D', 2.24, 24.8, 33.3), ('AVTrack', 3.11, 24.5, 32.4),
    ('AVTrack-D', 2.06, 25.4, 33.9), ('SGLATrack', 2.59, 24.9, 32.6), ('LETrack', 5.76, 27.5, 34.2)]
OFF = {'ORTrack': (-6, 0.6, 'right'), 'ORTrack-D': (0, -0.25, 'center'), 'SGLATrack': (0, 0.6, 'center'), 'AVTrack-D': (0, 0.75, 'center'), 'AVTrack': (-7, 0.15, 'right'), 'LETrack': (6, -0.2, 'left')}
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for i, (n, ms, a0, a1) in enumerate(models):
    f0, f1 = 1000 / ms, 1000 / (ms + CMC_MS)
    ax.annotate('', xy=(f1, a1), xytext=(f0, a0), arrowprops=dict(arrowstyle='->', color=GRID, lw=1))
    ax.scatter(f0, a0, s=28, color=BLUE, zorder=3, edgecolor=SURF, linewidth=1.5, label='Baseline' if i == 0 else None)
    ax.scatter(f1, a1, s=28, color=ORANGE, zorder=3, edgecolor=SURF, linewidth=1.5, label='+ CMC' if i == 0 else None)
    dx, dy, ha = OFF.get(n, (6, -0.3, 'left'))
    ax.text(f0 + dx, a0 + dy, n, fontsize=6.5, color=INK2, va='top', ha=ha)
ax.scatter(1000 / 8.04, 35.7, s=60, marker='*', color=AQUA, zorder=4, edgecolor=INK, linewidth=0.6,
           label='Ours (UAV+tiny FT + CMC)')
ax.text(1000 / 8.04 + 6, 35.7, '35.7', fontsize=7, color=INK, va='center')
ax.scatter(139, 30.9, s=28, marker='s', color=INK2, zorder=3, label='OSTrack ViT-B')
ax.set_xlabel('Speed (FPS, server GPU)'); ax.set_ylabel('BioDrone val AUC')
ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.5, loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_speed_acc.png'), dpi=300); plt.close(fig)
print(OUT)

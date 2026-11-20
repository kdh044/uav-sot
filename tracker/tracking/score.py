"""OPE 채점: pytracking calc_seq_err_robust 그대로 사용 (ORTrack analysis와 동일 규칙).

usage: python tools/score.py <result_dir> <dataset> [--frames out.npz]
  result_dir: <seq>.txt 가 있는 폴더 (우리 재현 결과 or 공식 raw 결과)
  dataset: dtb70 | uavdt | visdrone2018 | uav123
출력: Precision@20px, Success AUC (시퀀스 평균), 선택 시 프레임 단위 기록
"""
import argparse, os, sys
import numpy as np, torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from lib.test.evaluation import get_dataset
from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.utils.load_text import load_text

THR_OV = torch.arange(0.0, 1.05, 0.05, dtype=torch.float64)


def score(result_dir, dataset_name, frames_out=None):
    ds = get_dataset(dataset_name)
    prec, auc, rows = [], [], []
    for seq in ds:
        anno = torch.tensor(seq.ground_truth_rect, dtype=torch.float64)
        path = os.path.join(result_dir, seq.name + '.txt')
        pred = torch.tensor(load_text(path, delimiter=('\t', ','), dtype=np.float64))
        ov, ce, _, valid = calc_seq_err_robust(pred, anno, seq.dataset, None)
        n = anno.shape[0]
        auc.append(((ov.view(-1, 1) > THR_OV.view(1, -1)).sum(0).double() / n).mean().item())
        prec.append(((ce <= 20).sum().double() / n).item())
        if frames_out:
            size = torch.sqrt(anno[:, 2].clamp(min=0) * anno[:, 3].clamp(min=0))
            for i in range(1, n):  # 첫 프레임은 GT로 덮어써지므로 제외
                rows.append((seq.name, i, size[i].item(), ov[i].item(), ce[i].item(), bool(valid[i])))
    if frames_out:
        arr = np.array(rows, dtype=[('seq', 'U64'), ('frame', 'i4'), ('size', 'f8'),
                                    ('iou', 'f8'), ('cerr', 'f8'), ('valid', '?')])
        np.save(frames_out, arr)
    return 100 * np.mean(prec), 100 * np.mean(auc), len(ds)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('result_dir'); p.add_argument('dataset')
    p.add_argument('--frames', default=None)
    a = p.parse_args()
    pr, au, n = score(a.result_dir, a.dataset, a.frames)
    print(f'{a.dataset:13s} n={n:3d}  Prec@20 {pr:5.1f}  AUC {au:5.1f}')

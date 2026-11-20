import os
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList


class SatSOTDataset(BaseDataset):
    """SatSOT (위성 영상 소형 대상 SOT). <seq>/img/*.jpg, <seq>/groundtruth.txt (x,y,w,h; 'none' = 결측 → 0)"""

    def __init__(self):
        super().__init__()
        self.base_path = os.environ.get('SOT_SATSOT', '/home/airlab/user/kdh/datasets/sot/eval/satsot/SatSOT')
        self.sequence_list = sorted(d for d in os.listdir(self.base_path)
                                    if os.path.isdir(os.path.join(self.base_path, d)))

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, name):
        rows = []
        for line in open(os.path.join(self.base_path, name, 'groundtruth.txt')):
            line = line.strip()
            rows.append([0, 0, 0, 0] if (not line or line.lower() == 'none') else [float(v) for v in line.replace('\t', ',').split(',')[:4]])
        gt = np.array(rows, dtype=np.float64)
        fd = os.path.join(self.base_path, name, 'img')
        frames = sorted(os.path.join(fd, f) for f in os.listdir(fd) if f.endswith('.jpg'))
        n = min(len(frames), len(gt))
        # 첫 프레임 GT 가 결측이면 첫 유효 프레임부터 시작
        k = int(np.argmax((gt[:n, 2] > 0) & (gt[:n, 3] > 0)))
        return Sequence(name, frames[k:n], 'satsot', gt[k:n].reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

import json, os
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList


class SeaDronesSeeDataset(BaseDataset):
    """SeaDronesSee SOT val (해상, 4K). clip json: {clip: {"000267.png": ".../images/val/267.png"}}, GT val_annotations/<clip>.txt"""

    def __init__(self):
        super().__init__()
        self.base = '/home/airlab/user/kdh/datasets/sot/unused/seadronessee'
        self.clips = json.load(open(os.path.join(self.base, 'sot_annotations', 'SeaDronesSee_val.json')))
        self.sequence_list = sorted(self.clips, key=int)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, name):
        items = sorted(self.clips[name].items())
        frames = [os.path.join(self.base, 'images', 'val', os.path.basename(p).replace('.png', '.jpg')) for _, p in items]
        gt = np.loadtxt(os.path.join(self.base, 'sot_annotations', 'val_annotations', f'{name}.txt'), delimiter=',', ndmin=2)
        n = min(len(frames), len(gt))
        return Sequence('sds_' + name, frames[:n], 'seadronessee', gt[:n].reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

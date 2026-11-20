"""UAV SOT 학습용 데이터셋 (BioDrone train, VisDrone-SOT train).

둘 다 <root>/sequences/<seq>/*.jpg + 시퀀스별 x,y,w,h 텍스트 주석 구조.
주석이 0,0,0,0 (또는 w/h<=0) 인 프레임은 대상 없음(visible=False).
"""
import os
from collections import OrderedDict

import numpy as np
import torch

from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader


class _UAVSeqDataset(BaseVideoDataset):
    def __init__(self, name, root, anno_fn, frame_sort_key, exclude=None, image_loader=jpeg4py_loader):
        super().__init__(name, root, image_loader)
        seq_dir = os.path.join(root, 'sequences')
        exclude = set(exclude or [])
        self.sequence_list = sorted(s for s in os.listdir(seq_dir) if s not in exclude)
        self.anno_fn = anno_fn
        self.frame_sort_key = frame_sort_key
        self._frames = {}

    def get_name(self):
        return self.name

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return True

    def _frame_list(self, seq):
        if seq not in self._frames:
            d = os.path.join(self.root, 'sequences', seq)
            fs = [f for f in os.listdir(d) if f.endswith('.jpg')]
            fs.sort(key=self.frame_sort_key)
            self._frames[seq] = [os.path.join(d, f) for f in fs]
        return self._frames[seq]

    def get_sequence_info(self, seq_id):
        seq = self.sequence_list[seq_id]
        gt = np.loadtxt(self.anno_fn(self.root, seq), delimiter=',', dtype=np.float32).reshape(-1, 4)
        n = min(len(gt), len(self._frame_list(seq)))
        bbox = torch.tensor(gt[:n])
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        return {'bbox': bbox, 'valid': valid, 'visible': valid.clone().byte()}

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq = self.sequence_list[seq_id]
        frames = self._frame_list(seq)
        frame_list = [self.image_loader(frames[f]) for f in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        anno_frames = {k: [v[f, ...].clone() for f in frame_ids] for k, v in anno.items()}
        meta = OrderedDict({'object_class_name': None, 'motion_class': None, 'major_class': None,
                            'root_class': None, 'motion_adverb': None})
        return frame_list, anno_frames, meta


class BioDroneTrain(_UAVSeqDataset):
    """BioDrone train 뷰 (sequences/frame_XXX, anno/frame_XXX.txt). exclude: 튜닝 분할 시퀀스."""
    def __init__(self, root, exclude_dir=None, image_loader=jpeg4py_loader):
        exclude = os.listdir(os.path.join(exclude_dir, 'sequences')) if exclude_dir else None
        super().__init__('biodrone_train', root, lambda r, s: os.path.join(r, 'anno', s + '.txt'),
                         lambda f: int(f[3:-4]), exclude, image_loader)


class VisDroneSOTTrain(_UAVSeqDataset):
    """VisDrone-SOT train (sequences/<seq>/img*.jpg, annotations/<seq>.txt)."""
    def __init__(self, root, image_loader=jpeg4py_loader):
        super().__init__('visdrone_sot_train', root, lambda r, s: os.path.join(r, 'annotations', s + '.txt'),
                         lambda f: int(''.join(c for c in f if c.isdigit())), None, image_loader)


class LaTOTTrain(BaseVideoDataset):
    """LaTOT (소형 대상 추적, 434 시퀀스): <root>/<seq>/img/*.jpg + <root>/<seq>/<seq>.txt (탭/쉼표 x,y,w,h).
    공식 분할이 없고 평가셋이 아니므로 전부 학습에 사용. '*-deleted' 폴더 제외. 파싱 불가/w,h<=0 는 대상 없음."""
    def __init__(self, root, image_loader=jpeg4py_loader):
        super().__init__('latot', root, image_loader)
        self.sequence_list = sorted(s for s in os.listdir(root)
                                    if os.path.isdir(os.path.join(root, s)) and not s.endswith('-deleted')
                                    and os.path.exists(os.path.join(root, s, s + '.txt')))
        self._frames, self._anno = {}, {}

    def get_name(self):
        return self.name

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return True

    def _frame_list(self, seq):
        if seq not in self._frames:
            d = os.path.join(self.root, seq, 'img')
            self._frames[seq] = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith('.jpg')]
        return self._frames[seq]

    def _load_anno(self, seq):
        if seq not in self._anno:
            rows = []
            for line in open(os.path.join(self.root, seq, seq + '.txt')):
                try:
                    v = [float(x) for x in line.replace(',', ' ').split()[:4]]
                    rows.append(v if len(v) == 4 else [0, 0, 0, 0])
                except ValueError:
                    rows.append([0, 0, 0, 0])
            self._anno[seq] = np.array(rows, dtype=np.float32).reshape(-1, 4)
        return self._anno[seq]

    def get_sequence_info(self, seq_id):
        seq = self.sequence_list[seq_id]
        gt = self._load_anno(seq)
        n = min(len(gt), len(self._frame_list(seq)))
        bbox = torch.tensor(gt[:n])
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        return {'bbox': bbox, 'valid': valid, 'visible': valid.clone().byte()}

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq = self.sequence_list[seq_id]
        frames = self._frame_list(seq)
        frame_list = [self.image_loader(frames[f]) for f in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        anno_frames = {k: [v[f, ...].clone() for f in frame_ids] for k, v in anno.items()}
        meta = OrderedDict({'object_class_name': None, 'motion_class': None, 'major_class': None,
                            'root_class': None, 'motion_adverb': None})
        return frame_list, anno_frames, meta

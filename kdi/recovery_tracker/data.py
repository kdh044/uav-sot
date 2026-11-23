"""Frozen source-group diagnostic split and aligned images/GT/SAM3 cache.

These eight previously inspected TrainReal clips are not an independent test set.
No missing annotation is converted into an absence label.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from PIL import Image
import torch
from .losses import xywh_to_cxcywh

SOT = Path(__file__).resolve().parents[1]
PILOT = SOT / 'analysis/icml_20261120'
DEFAULT_MANIFEST = PILOT / 'birdsai_pilot_manifest.json'


def image_tensor(image):
    value = torch.from_numpy(np.array(image.convert('RGB'), copy=True)).permute(2,0,1).float()/255
    return (value-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]


def template_images(image, box):
    x,y,w,h = box
    side = max(2, math.ceil(math.sqrt(w*h)*2))
    cx,cy = x+w/2,y+h/2
    context = image.crop((round(cx-side/2),round(cy-side/2),round(cx-side/2)+side,round(cy-side/2)+side)).resize((128,128))
    detail = image.crop((x,y,x+w,y+h)).resize((128,128))
    return image_tensor(detail),image_tensor(context)


def diagnostic_partition(manifest):
    clips = manifest['pilot_clips']
    groups = sorted({c['source_group'] for c in clips}, key=lambda s: hashlib.sha256(('redesign-v1:'+s).encode()).hexdigest())
    if len(groups) < 4:
        raise ValueError('At least four source groups are required')
    return dict(train=groups[:-2], validation=groups[-2:], test=[],
                scope='Exploratory split of previously inspected thermal TRAIN clips; no independent test claim')


class DiagnosticPairs:
    def __init__(self, manifest_path=DEFAULT_MANIFEST, role='train', stride=15,
                 teacher_mode='filtered', mask_directory=None):
        self.path = Path(manifest_path)
        self.manifest_sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        manifest = json.loads(self.path.read_text())
        self.partition = diagnostic_partition(manifest)
        allowed = set(self.partition[role])
        self.clips = [c for c in manifest['pilot_clips'] if c['source_group'] in allowed]
        self.teacher_mode = teacher_mode
        self.mask_directory = Path(mask_directory) if mask_directory else PILOT/'pilot_outputs/sam3_masks'
        self.knowledge_directory = PILOT/'pilot_outputs/sam3_knowledge'
        self.samples,self.templates,self.teachers,self.priors = [],{},{},{}
        for ci,c in enumerate(self.clips):
            with Image.open(c['frames'][0]) as image:
                self.templates[ci] = template_images(image,c['gt'][0])
            for name,dest in [('sam3',self.teachers),('baseline',self.priors)]:
                path = PILOT/'pilot_outputs'/name/(c['name']+'.npz')
                with np.load(path,allow_pickle=False) as cache:
                    if str(cache['manifest_sha256']) != self.manifest_sha:
                        raise ValueError(f'Manifest/cache mismatch: {path}')
                    dest[ci] = cache['boxes'].copy()
            for i in range(1,len(c['frames']),stride):
                if np.isfinite(c['gt'][i]).all() and min(c['gt'][i][2:])>0:
                    self.samples.append((ci,i))
        self._mask_cache = {}
        self._knowledge_cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        ci,fi = self.samples[index]
        clip = self.clips[ci]
        with Image.open(clip['frames'][fi]) as image:
            w,h = image.size
            current = image_tensor(image)
        with Image.open(clip['frames'][fi-1]) as image:
            previous = image_tensor(image)
        scale = torch.tensor([w,h,w,h],dtype=torch.float32)
        prior = torch.tensor(self.priors[ci][fi-1],dtype=torch.float32)/scale
        gt = torch.tensor(clip['gt'][fi],dtype=torch.float32)/scale
        teacher = torch.tensor(self.teachers[ci][fi],dtype=torch.float32)/scale
        detail,context = self.templates[ci]
        mask_path = self.mask_directory/(clip['name']+'.npz')
        teacher_mask = None
        pointer = anchor_pointer = memory_descriptor = None
        knowledge_path = self.knowledge_directory/(clip['name']+'.npz')
        if knowledge_path.exists():
            mask_path = knowledge_path
        if mask_path.exists():
            if ci not in self._mask_cache:
                with np.load(mask_path,allow_pickle=False) as a:
                    if str(a['manifest_sha256']) != self.manifest_sha:
                        raise ValueError('Mask/manifest mismatch')
                    if not np.array_equal(a['frame_ids'],np.asarray(clip['frame_ids'])):
                        raise ValueError('Mask frame IDs do not match')
                    # Keep boxes and masks from the SAME causal SAM3 run.
                    self._mask_cache[ci] = (a['foreground'].copy(),a['boxes'].copy())
                    if 'object_pointer' in a:
                        self._knowledge_cache[ci] = (a['object_pointer'].copy(),a['memory_descriptor'].copy())
            masks,boxes = self._mask_cache[ci]
            teacher_mask = torch.tensor(masks[fi][None],dtype=torch.float32)
            teacher = torch.tensor(boxes[fi],dtype=torch.float32)/scale
            if ci in self._knowledge_cache:
                pointers,descriptors=self._knowledge_cache[ci]
                pointer=torch.tensor(pointers[fi:fi+1],dtype=torch.float32)
                anchor_pointer=torch.tensor(pointers[:1],dtype=torch.float32)
                memory_descriptor=torch.tensor(descriptors[fi:fi+1],dtype=torch.float32)
        if not torch.isfinite(prior).all() or (prior[2:]<=0).any():
            raise ValueError('Invalid past tracker state')
        return dict(current=current[None],previous=previous[None],template_detail=detail[None],
                    template_context=context[None],prior_box=xywh_to_cxcywh(prior)[None],
                    gt=xywh_to_cxcywh(gt)[None],teacher_box=xywh_to_cxcywh(teacher)[None],
                    teacher_mask=teacher_mask[None] if teacher_mask is not None else None,
                    teacher_pointer=pointer,teacher_anchor_pointer=anchor_pointer,
                    teacher_memory_descriptor=memory_descriptor,
                    key=f'{clip["name"]}:{fi}',source_group=clip['source_group'])

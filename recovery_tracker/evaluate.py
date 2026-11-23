"""Full-sequence diagnostic evaluation, current GT used only after inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import types
import numpy as np
from PIL import Image
import torch
from .data import DEFAULT_MANIFEST,diagnostic_partition
from .runtime import load_student,StudentRuntime


def run_sequence(tracker,frames,initial_box):
    boxes,scores,misses,writes,candidates=[],[],[],[],[]
    started=time.perf_counter()
    for i,path in enumerate(frames):
        with Image.open(path) as image:
            image=image.convert('RGB')
            if i==0:
                tracker.initialize(image,list(initial_box))
                boxes.append(initial_box);scores.append(1.);misses.append(False);writes.append(False)
                candidates.append(np.tile(initial_box,(8,1)))
            else:
                # Both original tracker and student expose target_bbox.
                out=tracker.track(image if isinstance(tracker,StudentRuntime) else np.asarray(image))
                boxes.append(out['target_bbox']);scores.append(out.get('score',np.nan))
                misses.append(out.get('no_match',False));writes.append(out.get('memory_written',False))
                cb=out.get('candidate_boxes',np.tile(out['target_bbox'],(8,1)))
                candidates.append(cb)
    torch.cuda.synchronize()
    return dict(boxes=np.asarray(boxes),score=np.asarray(scores),no_match=np.asarray(misses),
                memory_written=np.asarray(writes),candidates=np.asarray(candidates),seconds=time.perf_counter()-started)


def iou(a,b):
    lo=np.maximum(a[...,:2],b[...,:2]);hi=np.minimum(a[...,:2]+a[...,2:],b[...,:2]+b[...,2:])
    inter=np.maximum(hi-lo,0).prod(-1)
    return inter/np.maximum(a[...,2:].prod(-1)+b[...,2:].prod(-1)-inter,1e-9)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--joint-search',choices=['direct','confirm'])
    ap.add_argument('--global-recovery',action='store_true')
    ap.add_argument('--candidate-head',type=Path)
    ap.add_argument('--original-ostrack',action='store_true')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--role',choices=['train','validation','all'],default='validation')
    args=ap.parse_args()
    if args.original_ostrack==bool(args.checkpoint):ap.error('Choose checkpoint OR original-ostrack')
    if args.candidate_head and args.original_ostrack:ap.error('--candidate-head requires a student checkpoint')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(43)
    manifest=json.loads(args.manifest.read_text());partition=diagnostic_partition(manifest)
    if args.original_ostrack:
        source=Path(__file__).resolve().parents[1]/'research_bases/OSTrack'
        sys.path.insert(0,str(source))
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        # Process-local compatibility for an unused legacy training-loader import.
        # Keep the official worktree and original tracking decisions unchanged.
        if 'torch._six' not in sys.modules:
            shim=types.ModuleType('torch._six')
            shim.string_classes=(str,bytes)
            shim.int_classes=int
            sys.modules['torch._six']=shim
        from lib.config.ostrack.config import cfg,update_config_from_file
        from lib.test.tracker.ostrack import OSTrack
        from .student import DEFAULT_CONTEXT
        update_config_from_file(str(source/'experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml'))
        params=SimpleNamespace(cfg=cfg,checkpoint=str(DEFAULT_CONTEXT),debug=0,save_all_boxes=False,
                               template_factor=2.,template_size=128,search_factor=4.,search_size=256)
        tracker=OSTrack(params,'birdsai')
        # The original API initializes from a NumPy image rather than PIL.
        initialize=tracker.initialize
        tracker.initialize=lambda im,box:initialize(np.asarray(im),{'init_bbox':box})
        condition='original_ostrack256_ce_margin10'
        train_groups=[]
    else:
        model,ckpt=load_student(args.checkpoint)
        if ckpt['protocol']['manifest_sha256']!=hashlib.sha256(args.manifest.read_bytes()).hexdigest():
            raise ValueError('Training/evaluation manifest mismatch')
        head=torch.load(args.candidate_head,map_location='cuda',weights_only=True) if args.candidate_head else None
        tracker=StudentRuntime(model,candidate_head=head,global_recovery=args.global_recovery,joint_search=args.joint_search)
        condition=ckpt['condition'] + ('+candidate_head' if head is not None else '') + ('+global_recovery' if args.global_recovery else '')
        train_groups=ckpt['protocol']['partition']['train']
        if args.joint_search:
            condition += '+joint_'+args.joint_search
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for clip in manifest['pilot_clips']:
        if args.role!='all' and clip['source_group'] not in partition[args.role]:continue
        if args.role=='validation' and clip['source_group'] in train_groups:
            raise ValueError('Training source leaked into validation')
        destination=args.out/(clip['name']+'.npz')
        # No later GT is provided to the inference function.
        result=run_sequence(tracker,clip['frames'],clip['gt'][0])
        gt=np.asarray(clip['gt']);valid=np.isfinite(gt).all(1)&(gt[:,2:]>0).all(1);valid[0]=False
        overlap=iou(result['boxes'],gt)
        oracle=iou(result['candidates'],gt[:,None]).max(-1)
        row=dict(name=clip['name'],source_group=clip['source_group'],valid_frames=int(valid.sum()),
                 auc=float((overlap[valid,None]>np.linspace(0,1,21)).mean()),
                 failure_fraction=float((overlap[valid]<.1).mean()),
                 candidate_recall50=float((oracle[valid]>=.5).mean()),
                 oracle_auc=float((oracle[valid,None]>np.linspace(0,1,21)).mean()),
                 no_match_fraction=float(result['no_match'][valid].mean()),
                 memory_writes=int(result['memory_written'].sum()),seconds=result['seconds'])
        np.savez_compressed(destination,**result,gt=gt,manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest())
        rows.append(row);print(json.dumps(row),flush=True)
    report=dict(condition=condition,role=args.role,scope=partition['scope'],sequences=rows,
                sequence_macro_auc=float(np.mean([r['auc'] for r in rows])),
                teacher_used_at_inference=False,first_gt_box_only=True,
                latency_scope='end-to-end sequence including image reads and initialization, excluding model load',
                limitations=['Two previously inspected thermal TrainReal source groups; not independent RGB/OPE benchmark.',
                             'Prototype state/write threshold=.5 is uncalibrated.',
                             'Training used frozen baseline past states; student-history distribution shift remains.'])
    (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print('MACRO_AUC',report['sequence_macro_auc'],flush=True)


if __name__=='__main__':main()

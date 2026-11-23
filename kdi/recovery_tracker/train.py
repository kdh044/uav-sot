"""Matched small diagnostic training. Not a publication training protocol.

Example: PYTHONPATH=SOT:SOT/pydeps python -m recovery_tracker.train --conditions gt output observation
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from .data import DiagnosticPairs,DEFAULT_MANIFEST
from .student import RecoveryStudent,DEFAULT_CONTEXT
from .losses import tracking_loss, pair_iou

SOT=Path(__file__).resolve().parents[1]


def move(sample,device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in sample.items()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    ap.add_argument('--out',type=Path,default=SOT/'analysis/redesign_20261120/training_v1')
    ap.add_argument('--conditions',nargs='+',choices=['gt','output','observation','knowledge','shared','attention','coordinates'],default=['coordinates'])
    ap.add_argument('--steps',type=int,default=240)
    ap.add_argument('--stride',type=int,default=15)
    ap.add_argument('--seed',type=int,default=43)
    ap.add_argument('--dim',type=int,default=64)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--negative-fraction',type=float,default=.15)
    ap.add_argument('--native-proposals',action='store_true',help='Also propose queries from target-conditioned native detail foreground')
    ap.add_argument('--single-pair', type=int, help='Repeat one training pair for an overfit diagnostic')
    ap.add_argument('--prior-shift',type=float,default=0.,
                    help='Maximum normalized shift of the previous-box center during training')
    args=ap.parse_args()
    if not 0 <= args.prior_shift <= .5:
        ap.error('--prior-shift must be between 0 and .5')
    if args.steps<1:ap.error('steps must be positive')
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(False) # CUDA grid_sample backward may be nondeterministic.
    data=DiagnosticPairs(args.manifest,'train',args.stride)
    repaired_conditions = {'shared','attention','coordinates'}
    repaired = bool(repaired_conditions.intersection(args.conditions))
    knowledge = repaired or 'knowledge' in args.conditions
    if knowledge:
        absent=[c['name'] for c in data.clips if not (data.knowledge_directory/(c['name']+'.npz')).exists()]
        if absent:raise FileNotFoundError(f'Actual SAM3 pointer/memory export is required first: {absent}')
    rng=random.Random(args.seed)
    order=[]
    for step in range(args.steps):
        index=rng.randrange(len(data))
        negative=rng.random()<args.negative_fraction
        other=None
        if negative:
            source=data.clips[data.samples[index][0]]['source_group']
            eligible=[i for i,(ci,_) in enumerate(data.samples) if data.clips[ci]['source_group']!=source]
            other=rng.choice(eligible)
        order.append((index,other))
    if args.single_pair is not None:
        order = [(args.single_pair,None)] * args.steps
    context_cache={}
    manifest_sha=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    protocol=dict(scope='Exploratory thermal TrainReal diagnostic; previously inspected clips, no independent test',
                  partition=data.partition,manifest_sha256=manifest_sha,steps=args.steps,stride=args.stride,seed=args.seed,
                  order=order,teacher_min_iou=.5,teacher_better_margin=.05,kd_weight=.3,
                  input_prior='previous official ORTrack rollout box, current/future GT excluded from model inputs',
                  negatives='cross-source mismatched template/current pairs; assumed different identity, not annotation-derived absence',
                  reproducibility='same seed, initial weights, optimizer and order across conditions; CUDA grid_sample backward not bitwise deterministic')
    protocol['knowledge_distillation']=knowledge
    protocol['native_proposals']=args.native_proposals or repaired
    protocol['repaired_observation']=repaired
    protocol['single_pair']=args.single_pair
    protocol['prior_shift']=args.prior_shift
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    base=RecoveryStudent(dim=args.dim,knowledge_distillation=knowledge,native_proposals=args.native_proposals or repaired,repaired_observation=repaired).to(args.device)
    context_hash=hashlib.sha256(DEFAULT_CONTEXT.read_bytes()).hexdigest()
    # Save only trainable student components, never duplicate the frozen context backbone.
    initial={k:v.detach().cpu().clone() for k,v in base.state_dict().items() if not k.startswith('context.')}
    context=base.context
    (args.out/'initial_sha256.json').write_text(json.dumps({k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in initial.items()},indent=2)+'\n')
    del base
    for condition in args.conditions:
        dest=args.out/condition
        dest.mkdir(exist_ok=True)
        if (dest/'checkpoint.pt').exists():
            raise FileExistsError(f'Refusing to overwrite a completed condition: {dest}')
        torch.manual_seed(args.seed);np.random.seed(args.seed)
        model=RecoveryStudent(dim=args.dim,load_context=False,knowledge_distillation=knowledge,native_proposals=args.native_proposals or repaired,repaired_observation=repaired).to(args.device)
        model.context=context
        model.load_state_dict(initial,strict=False)
        model.train()
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=3e-4,weight_decay=1e-4)
        started=time.time()
        with (dest/'metrics.jsonl').open('w') as log:
            for step,(index,other) in enumerate(order):
                sample=data[index]
                if other is not None:
                    donor=data[other]
                    for k in ['current','previous','prior_box']:
                        sample[k]=donor[k]
                    sample['teacher_box']=torch.zeros_like(sample['teacher_box'])
                    sample['teacher_mask']=None
                sample=move(sample,args.device)
                if (condition in repaired_conditions or condition=='knowledge') and other is None and sample['teacher_pointer'] is None:
                    raise ValueError('Refusing to call output-only training knowledge distillation')
                cache_key=f'{index}:{other}'
                if cache_key not in context_cache:
                    context_cache[cache_key]=context(sample['template_context'],sample['current']).detach().cpu()
                kwargs={k:sample[k] for k in ['current','previous','template_detail','template_context','prior_box']}
                if args.prior_shift and other is None:
                    perturb = random.Random(args.seed*1000003+step)
                    if perturb.random() < .5:
                        state = kwargs['prior_box'].clone()
                        axis = perturb.randrange(2)
                        state[:,axis] = (state[:,axis] + perturb.uniform(-args.prior_shift,args.prior_shift)).clamp(0,1)
                        kwargs['prior_box'] = state
                output=model(**kwargs,context_features=context_cache[cache_key].to(args.device))
                present=torch.tensor([other is None],device=args.device)
                loss,metrics=tracking_loss(output,sample['gt'],kwargs['prior_box'],
                    sample['teacher_box'],sample['teacher_mask'],kd_weight=0 if condition=='gt' else .3,
                    observation_kd=condition in ['observation','knowledge'],target_present=present,
                    sampling_mode=condition if condition in repaired_conditions else None,
                    teacher_pointer=sample['teacher_pointer'] if condition in repaired_conditions or condition=='knowledge' else None,
                    teacher_anchor_pointer=sample['teacher_anchor_pointer'] if condition in repaired_conditions or condition=='knowledge' else None,
                    teacher_memory_descriptor=sample['teacher_memory_descriptor'] if condition in repaired_conditions or condition=='knowledge' else None)
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at {step}')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradients')
                optimizer.step()
                with torch.no_grad():
                    final = output['final']
                    ious = pair_iou(final['boxes_cxcywh'],sample['gt'][:,None])[0]
                    choice = int(final['association_logits'][0].argmax())
                    metrics.update(oracle_iou=float(ious.max()), no_match=choice==len(ious),
                                   selected_iou=float(ious[choice]) if choice<len(ious) else 0.)
                metrics.update(step=step,key=sample['key'],negative=other is not None,
                               teacher_mask=sample['teacher_mask'] is not None,gradient_norm=float(norm))
                log.write(json.dumps(metrics)+'\n');log.flush()
                if step%20==0:print(condition,step,round(metrics['total'],4),round(time.time()-started,1),flush=True)
        state={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('context.')}
        torch.save(dict(student=state,config=model.config,condition=condition,protocol=protocol,
                        context_checkpoint=str(DEFAULT_CONTEXT),context_sha256=context_hash,
                        seconds=time.time()-started,trained_steps=args.steps,
                        validation_status='training completed; not evaluated yet'),dest/'checkpoint.pt')
        print('DONE',condition,round(time.time()-started,2),flush=True)


if __name__=='__main__':main()

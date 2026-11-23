"""Separate one-step candidate quality from closed-loop state drift.

This uses past baseline boxes, NOT past/current GT, and is NOT OPE tracking.
"""
import argparse
import json
from pathlib import Path
import torch
from .data import DiagnosticPairs
from .runtime import load_student
from .train import move
from .losses import pair_iou


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--stride',type=int,default=15)
    args=ap.parse_args();torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data=DiagnosticPairs(role='validation',stride=args.stride)
    model,checkpoint=load_student(args.checkpoint)
    rows=[]
    with torch.no_grad():
        for i in range(len(data)):
            s=move(data[i],'cuda')
            output=model(**{k:s[k] for k in ['current','previous','template_detail','template_context','prior_box']})
            p=output['final'];u=pair_iou(p['boxes_cxcywh'],s['gt'][:,None])[0]
            choice=int(p['association_logits'][0].argmax())
            rows.append(dict(key=s['key'],no_match=choice==len(u),oracle_iou=float(u.max()),
                             selected_iou=float(u[choice]) if choice<len(u) else 0.))
    summary=dict(condition=checkpoint['condition'],scope='One-step validation with previous baseline boxes; NOT continuous tracking or GT teacher forcing',
                 samples=len(rows),oracle_recall50=sum(r['oracle_iou']>=.5 for r in rows)/len(rows),
                 oracle_recall10=sum(r['oracle_iou']>=.1 for r in rows)/len(rows),
                 no_match_fraction=sum(r['no_match'] for r in rows)/len(rows),rows=rows)
    args.out.write_text(json.dumps(summary,indent=2)+'\n');print({k:v for k,v in summary.items() if k!='rows'})


if __name__=='__main__':main()

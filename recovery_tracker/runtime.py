"""Causal first-box-only student runtime. SAM3 and later GT are not inputs."""
import hashlib
from pathlib import Path
import torch
from .data import image_tensor,template_images
from .student import RecoveryStudent,sample_box_feature
from .losses import xywh_to_cxcywh


def load_student(path,device='cuda'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    context=Path(checkpoint['context_checkpoint'])
    if hashlib.sha256(context.read_bytes()).hexdigest()!=checkpoint['context_sha256']:
        raise ValueError('Frozen context checkpoint changed')
    model=RecoveryStudent(**checkpoint['config'],context_checkpoint=context).to(device)
    missing,unexpected=model.load_state_dict(checkpoint['student'],strict=False)
    if unexpected or any(not key.startswith('context.') for key in missing):
        raise ValueError(f'Student checkpoint mismatch: {missing}, {unexpected}')
    return model.eval(),checkpoint


class StudentRuntime:
    def __init__(self,model,device='cuda',write_threshold=.5, candidate_head=None, global_recovery=False, joint_search=None, joint_scorer=None):
        self.model,self.device,self.write_threshold=model,device,write_threshold


        self.candidate_head = candidate_head
        self.global_recovery = global_recovery
        self.joint_search = joint_search
        self.joint_scorer = joint_scorer

    @torch.no_grad()
    def initialize(self,image,initial_box):
        w,h=image.size
        detail,context=template_images(image,initial_box)
        self.template_detail,self.template_context=detail[None].to(self.device),context[None].to(self.device)
        self.previous=image_tensor(image)[None].to(self.device)
        scale=torch.tensor([w,h,w,h],device=self.device)
        self.state=xywh_to_cxcywh(torch.tensor(initial_box,device=self.device)/scale)[None]
        anchor,_=self.model.detail(self.template_detail)
        self.anchor=anchor.mean((2,3))
        self.recent=self.anchor.clone()
        if self.model.config.get('repaired_observation'):
            fine,_=self.model.detail(self.previous)
            self.recent=self.model.encode_memory(fine,self.state,self.previous.shape[-2:])
        self.pending_recovery = None
        if self.global_recovery or self.joint_search:
            import numpy as np
            from .global_search import make_template
            self.first_patch = make_template(np.asarray(image.convert('L')), initial_box)
        return list(initial_box)

    @torch.no_grad()
    def track(self,image):
        w,h=image.size
        current=image_tensor(image)[None].to(self.device)
        output=self.model(current,self.previous,self.template_detail,self.template_context,
                          self.state,memory=torch.stack([self.anchor,self.recent],1))
        final=output['final']
        probabilities=final['association_logits'][0].softmax(-1)
        choice=int(probabilities.argmax())
        no_match=choice==self.model.config['queries']
        score=float(probabilities[:-1].max())
        if self.candidate_head is not None:
            head = self.candidate_head
            features = torch.cat([output['query_features'][0], final['write_logits'][0,:,None]], -1)
            rank, accept = torch.nn.functional.linear(
                (features-head['shift'])/head['scale'],
                head['head']['weight'],head['head']['bias']).unbind(-1)
            valid = final['sample_valid'][0].flatten(1).any(-1)
            choice = int(rank.masked_fill(~valid,-torch.inf).argmax())
            score = float(accept[choice].sigmoid())
            no_match = not bool(valid[choice]) or score < head['threshold']
        candidates=final['boxes_cxcywh'][0]
        recovered = False
        pixel_scale = candidates.new_tensor([w,h,w,h])
        if self.joint_search:
            import numpy as np
            from .global_search import overlap, patch_scores, search
            gray = np.asarray(image.convert('L'))
            local = (torch.cat([candidates[:,:2]-candidates[:,2:]/2,candidates[:,2:]],-1)
                     * pixel_scale).cpu().numpy()
            valid = final['sample_valid'][0].flatten(1).any(-1).cpu().numpy()
            if self.joint_scorer is None:
                global_boxes = search(gray, *self.first_patch)
                pool = np.concatenate([local[valid],global_boxes])
                scores = patch_scores(gray,self.first_patch[0],pool)
            else:
                pool, scores = self.joint_scorer(gray, self.first_patch, local[valid])
            best = int(scores.argmax())
            proposal = pool[best]
            confirmed = (self.pending_recovery is not None and
                         overlap(proposal,self.pending_recovery) >= .3)
            self.pending_recovery = proposal if np.isfinite(scores[best]) else None
            score = float(scores[best]) if self.pending_recovery is not None else 0.
            recovered = self.pending_recovery is not None and (self.joint_search == 'direct' or confirmed)
            no_match = not recovered
            if recovered:
                box = xywh_to_cxcywh(candidates.new_tensor(proposal)/pixel_scale)
                write = False
        elif no_match and self.global_recovery:
            import numpy as np
            from .global_search import overlap, search
            global_boxes = search(np.asarray(image.convert('L')), *self.first_patch)
            proposal = global_boxes[0] if len(global_boxes) else None
            if proposal is not None and self.pending_recovery is not None:
                recovered = bool(overlap(proposal, self.pending_recovery) >= .3)
            self.pending_recovery = None if recovered else proposal
            if recovered:
                box = xywh_to_cxcywh(candidates.new_tensor(proposal)/pixel_scale)
                write = False
                no_match = False
        else:
            self.pending_recovery = None
        if no_match:
            box=self.state[0].clone()
            box[:2]+=output['displacement'][0]
            write=False
        elif not recovered:
            box=candidates[choice].clone()
            write=bool(final['write_logits'][0,choice].sigmoid()>self.write_threshold)
            if write:
                # Store observed appearance, not GT or a teacher feature.
                self.recent=self.model.encode_memory(output['detail_feature'],box[None],current.shape[-2:]).detach()
        lo=box[:2]-box[2:]/2
        hi=box[:2]+box[2:]/2
        minimum=box.new_tensor([1/w,1/h])
        lo=torch.minimum(lo.clamp_min(0),1-minimum)
        hi=torch.maximum(torch.minimum(hi,torch.ones_like(hi)),lo+minimum)
        self.state=torch.cat([(lo+hi)/2,hi-lo])[None]
        self.previous=current
        scale=box.new_tensor([w,h,w,h])
        xywh=torch.cat([lo,hi-lo])*scale
        candidate_xywh=torch.cat([candidates[:,:2]-candidates[:,2:]/2,candidates[:,2:]],-1)*scale
        return dict(target_bbox=xywh.cpu().tolist(),score=score,
                    no_match=no_match,global_recovered=recovered,memory_written=write,candidate_boxes=candidate_xywh.cpu().numpy())

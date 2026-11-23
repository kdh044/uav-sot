"""GT-supervised tracking and optional quality-filtered SAM3 output distillation."""
import torch
from torch.nn import functional as F
from .geometry import feature_centers, resample_uniform


def xywh_to_cxcywh(boxes):
    return torch.cat([boxes[..., :2] + boxes[..., 2:] * .5, boxes[..., 2:]], -1)


def pair_iou(a, b):
    alo, ahi = a[..., :2] - a[..., 2:] * .5, a[..., :2] + a[..., 2:] * .5
    blo, bhi = b[..., :2] - b[..., 2:] * .5, b[..., :2] + b[..., 2:] * .5
    overlap = (torch.minimum(ahi, bhi) - torch.maximum(alo, blo)).clamp_min(0).prod(-1)
    union = a[..., 2:].clamp_min(0).prod(-1) + b[..., 2:].clamp_min(0).prod(-1) - overlap
    return overlap / union.clamp_min(1e-9)


def giou(a, b):
    alo, ahi = a[..., :2] - a[..., 2:] * .5, a[..., :2] + a[..., 2:] * .5
    blo, bhi = b[..., :2] - b[..., 2:] * .5, b[..., :2] + b[..., 2:] * .5
    inter = (torch.minimum(ahi, bhi) - torch.maximum(alo, blo)).clamp_min(0).prod(-1)
    union = a[..., 2:].prod(-1) + b[..., 2:].prod(-1) - inter
    cover = (torch.maximum(ahi, bhi) - torch.minimum(alo, blo)).clamp_min(0).prod(-1)
    return inter / union.clamp_min(1e-9) - (cover - union) / cover.clamp_min(1e-9)


def balanced_bce(logits, target):
    # Soft teacher foreground occupancy is preserved; do not threshold away tiny targets.
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    positive = (raw * target).sum() / target.sum().clamp_min(1)
    negative = (raw * (1 - target)).sum() / (1 - target).sum().clamp_min(1)
    return positive + negative


def center_target(box, shape):
    h, w = shape
    y = (torch.arange(h, device=box.device) + .5) / h
    x = (torch.arange(w, device=box.device) + .5) / w
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    size = torch.maximum(box[:, 2:] * .4, box.new_tensor([.75 / w, .75 / h]))
    distance = ((xx[None] - box[:, 0, None, None]) / size[:, 0, None, None]).square()
    distance = distance + ((yy[None] - box[:, 1, None, None]) / size[:, 1, None, None]).square()
    return (-.5 * distance).exp()[:, None]


def tracking_loss(outputs, gt, prior_box, teacher_box=None, teacher_mask=None,
                  kd_weight=0., teacher_min_iou=.5, target_present=None,
                  observation_kd=False, teacher_better_margin=.05,
                  sampling_mode=None, teacher_pointer=None,teacher_anchor_pointer=None,teacher_memory_descriptor=None):
    """Inputs are labelled training frames only. Missing annotations must be excluded.

    GT gates the TRAINING teacher only; no GT-dependent gate exists at inference.
    Empty teachers are unknown and never supervise no-match. Teacher box agreement
    is necessary here but does not guarantee mask or identity correctness.
    """
    b = gt.shape[0]
    if target_present is None:
        target_present = torch.ones(b, device=gt.device, dtype=torch.bool)
    positive = target_present.bool()
    zero = outputs['final']['boxes_cxcywh'].sum() * 0
    losses = {}
    for layer_index, prediction in enumerate(outputs['auxiliary'] + [outputs['final']]):
        boxes = prediction['boxes_cxcywh']
        ious = pair_iou(boxes, gt[:, None])
        cost = 2 * (boxes.detach() - gt[:, None]).abs().mean(-1) - ious.detach()
        visible = prediction['sample_valid'].flatten(2).any(-1)
        if 'native_stride' in outputs:
            cost = cost.masked_fill(~visible & visible.any(-1, keepdim=True), float('inf'))
        index = cost.argmin(-1)
        chosen = boxes[torch.arange(b, device=gt.device), index]
        losses[f'box_{layer_index}'] = (5 * F.l1_loss(chosen[positive], gt[positive]) +
            2 * (1 - giou(chosen[positive], gt[positive])).mean()) if positive.any() else zero
        index = torch.where(positive, index, boxes.shape[1])
        eligible = ~positive | visible.any(-1) if 'native_stride' in outputs else torch.ones_like(positive)
        losses[f'identity_{layer_index}'] = F.cross_entropy(prediction['association_logits'][eligible], index[eligible]) if eligible.any() else zero
        quality_target = torch.where(positive[:, None], ious.detach(), torch.zeros_like(ious))
        losses[f'quality_{layer_index}'] = F.binary_cross_entropy_with_logits(prediction['write_logits'], quality_target)
    proposal_target = center_target(gt, outputs['proposal_logits'].shape[-2:]) * positive[:, None, None, None]
    losses['proposal'] = balanced_bce(outputs['proposal_logits'], proposal_target)
    # Weak box-derived foreground for GT-only training; real mask shape is a teacher signal.
    fh, fw = outputs['foreground_logits'].shape[-2:]
    # Exact fractional cell occupancy: a sub-stride target must not disappear.
    ex = torch.arange(fw+1, device=gt.device)/fw
    ey = torch.arange(fh+1, device=gt.device)/fh
    lo, hi = gt[:,:2]-gt[:,2:]/2, gt[:,:2]+gt[:,2:]/2
    ox = (torch.minimum(ex[None,1:],hi[:,0,None])-torch.maximum(ex[None,:-1],lo[:,0,None])).clamp_min(0)*fw
    oy = (torch.minimum(ey[None,1:],hi[:,1,None])-torch.maximum(ey[None,:-1],lo[:,1,None])).clamp_min(0)*fh
    weak = oy[:,:,None]*ox[:,None,:]*positive[:,None,None]
    if 'native_stride' in outputs:
        ih, iw = outputs['image_shape']
        stride = outputs['native_stride']
        cells = gt.new_tensor([stride / iw, stride / ih])
        centers = feature_centers(torch.arange(fh*fw, device=gt.device), (fh,fw), (ih,iw), stride)
        overlap = (torch.minimum(centers + cells/2, hi[:,None]) -
                   torch.maximum(centers - cells/2, lo[:,None])).clamp_min(0)
        weak = (overlap / cells).prod(-1).reshape(b,fh,fw) * positive[:,None,None]
        nearest = ((centers - gt[:,None,:2])/cells).square().sum(-1).argmin(-1)
        geometry = outputs['proposal_geometry'].flatten(2).transpose(1,2)
        rows = torch.arange(b, device=gt.device)
        prediction = geometry[rows,nearest]
        offset = (gt[:,:2]-centers[nearest])/cells
        scale = (gt[:,2:].clamp_min(1e-8)/prior_box[:,2:].clamp_min(1e-8)).log().clamp(-1,1)
        losses['proposal_geometry'] = F.smooth_l1_loss(prediction[positive].tanh(),
            torch.cat([offset,scale],-1)[positive]) if positive.any() else zero
    losses['foreground_gt_weak'] = balanced_bce(outputs['foreground_logits'], weak[:,None])
    mean = prior_box[:, :2] + outputs['displacement']
    sigma = outputs['motion_sigma']
    # Relative constant in Gaussian NLL keeps this diagnostic term easier to inspect.
    motion = .5*((gt[:,:2]-mean)/sigma).square() + torch.log(sigma/.005)
    losses['motion'] = .05 * motion[positive].mean() if positive.any() else zero
    losses['kd_box'] = gt.new_zeros(())
    losses['kd_identity'] = gt.new_zeros(())
    losses['kd_mask'] = gt.new_zeros(())
    losses['kd_observation'] = gt.new_zeros(())
    losses['kd_proposal'] = gt.new_zeros(())
    losses['kd_pointer'] = gt.new_zeros(())
    losses['kd_memory'] = gt.new_zeros(())
    losses['kd_temporal_relation'] = gt.new_zeros(())
    accepted = gt.new_zeros((b,), dtype=torch.bool)
    better_count = observation_supported = 0
    if kd_weight and teacher_box is not None:
        nonempty = (teacher_box[:, 2:] > 0).all(-1) & torch.isfinite(teacher_box).all(-1)
        quality = pair_iou(teacher_box, gt).detach()
        accepted = nonempty & positive & (quality >= teacher_min_iou)
        if accepted.any():
            boxes = outputs['final']['boxes_cxcywh'][accepted]
            teacher = teacher_box[accepted].detach()
            affinity = pair_iou(boxes.detach(), teacher[:, None])
            valid_candidates = outputs['final']['sample_valid'][accepted].flatten(2).any(-1)
            if 'native_stride' in outputs:
                proximity = (boxes.detach()[...,:2]-teacher[:,None,:2]).abs().sum(-1)
                score = affinity - .01 * proximity
                best = score.masked_fill(~valid_candidates & valid_candidates.any(-1, keepdim=True), -float('inf')).argmax(-1)
            else:
                best = affinity.argmax(-1)
            selected = boxes[torch.arange(len(boxes), device=gt.device), best]
            if teacher_pointer is not None:
                if 'distilled_queries' not in outputs:
                    raise ValueError('Teacher representations require the student knowledge-transfer heads')
                row = torch.arange(len(boxes),device=gt.device)
                sq=F.normalize(outputs['distilled_queries'][accepted][row,best],dim=-1)
                tq=F.normalize(teacher_pointer[accepted].detach(),dim=-1)
                losses['kd_pointer']=kd_weight*(1-(sq*tq).sum(-1)).mean()
                if teacher_anchor_pointer is not None:
                    sa=F.normalize(outputs['distilled_anchor'][accepted],dim=-1)
                    ta=F.normalize(teacher_anchor_pointer[accepted].detach(),dim=-1)
                    losses['kd_temporal_relation']=kd_weight*F.mse_loss((sq*sa).sum(-1),(tq*ta).sum(-1))
                if teacher_memory_descriptor is not None:
                    sm=F.normalize(outputs['distilled_memory'][accepted],dim=-1)
                    if sm.ndim == 3:
                        sm = sm[row,best]
                    tm=F.normalize(teacher_memory_descriptor[accepted].detach(),dim=-1)
                    losses['kd_memory']=kd_weight*(1-(sm*tm).sum(-1)).mean()
            student_best = pair_iou(boxes.detach(),gt[accepted,None]).max(-1).values
            better = quality[accepted] > student_best + teacher_better_margin
            better_count = int(better.sum())
            if better.any():
                losses['kd_box'] = kd_weight * (F.l1_loss(selected[better], teacher[better]) + (1-giou(selected[better], teacher[better])).mean())
            # Only teach correspondence when a student candidate actually overlaps teacher.
            if 'native_stride' in outputs:
                affinity = affinity.masked_fill(~valid_candidates, -1e4)
            supported = affinity.max(-1).values >= .1
            if supported.any():
                soft = (affinity[supported] / .15).softmax(-1)
                candidate_logits = outputs['final']['association_logits'][accepted][supported, :-1]
                losses['kd_identity'] = kd_weight * F.kl_div(candidate_logits.log_softmax(-1), soft, reduction='batchmean')
            if teacher_mask is not None:
                mask = F.interpolate(teacher_mask[accepted].detach().float(), (fh, fw), mode='area')
                raw_mask = teacher_mask[accepted].detach().float()
                native_mask = resample_uniform(raw_mask,(fh,fw),outputs['image_shape'],4) if 'native_stride' in outputs else mask
                losses['kd_mask'] = kd_weight * balanced_bce(outputs['foreground_logits'][accepted], native_mask)
                if observation_kd or sampling_mode is not None:
                    # Dense target distribution guides proposals even if sparse points miss target.
                    target = F.interpolate(mask,outputs['proposal_logits'].shape[-2:],mode='area').flatten(1)
                    nonzero = target.sum(-1)>1e-6
                    if nonzero.any():
                        target = target[nonzero]/target[nonzero].sum(-1,keepdim=True)
                        logits = outputs['proposal_logits'][accepted][nonzero].flatten(1)
                        losses['kd_proposal'] = kd_weight * F.kl_div(logits.log_softmax(-1),target,reduction='batchmean')
                    prediction = outputs['final']
                    row = torch.arange(len(boxes),device=gt.device)
                    locations = prediction['sample_locations'][accepted][row,best]
                    weights = prediction['sample_weights'][accepted][row,best]
                    valid = prediction['sample_valid'][accepted][row,best]
                    if sampling_mode in ('attention', 'coordinates'):
                        losses['kd_observation'] = kd_weight * observation_distance(
                            locations, weights, valid, raw_mask, sampling_mode == 'coordinates')
                    grid = locations.reshape(len(boxes),1,-1,2)*2-1
                    # Keep coordinate gradients here: teacher evidence supervises where to read.
                    evidence = F.grid_sample(mask,grid,align_corners=False)[:,0,0]
                    weights = weights.flatten(1)*valid.flatten(1)
                    support = (evidence*weights).sum(-1)
                    useful = support>1e-6
                    observation_supported = int(useful.sum())
                    if useful.any() and sampling_mode is None:
                        losses['kd_observation'] = kd_weight * -support[useful].log().mean()
    total = sum(losses.values())
    metrics = {key: float(value.detach()) for key, value in losses.items()}
    metrics.update(total=float(total.detach()), teacher_accepted=int(accepted.sum()),
                   teacher_better=better_count,observation_supported=observation_supported,labelled_frames=b)
    return total, metrics


def observation_distance(locations, weights, valid, mask, coordinate_grad):
    """Mask centroid distance supplies a gradient even with zero foreground overlap."""
    h, w = mask.shape[-2:]
    y = (torch.arange(h, device=mask.device) + .5) / h
    x = (torch.arange(w, device=mask.device) + .5) / w
    mass = mask[:, 0].sum((1, 2))
    usable = mass > 1e-6
    if not usable.any():
        return weights.sum() * 0
    center = torch.stack([(mask[:,0] * x).sum((1,2)),
                          (mask[:,0] * y[:,None]).sum((1,2))], -1) / mass.clamp_min(1e-6)[:,None]
    points = locations if coordinate_grad else locations.detach()
    distance = (points - center[:,None,None]).square().sum(-1)
    # A fully invalid query still gets a geometric signal in coordinate mode.
    importance = torch.where(valid.flatten(1).any(-1)[:,None,None],
                             weights, torch.full_like(weights, 1 / (weights.shape[-1]*weights.shape[-2])))
    return (distance * importance).sum((1,2))[usable].mean()

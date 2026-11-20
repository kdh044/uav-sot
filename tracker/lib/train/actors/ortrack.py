from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class ORTrackActor(BaseActor):
    """ Actor for training ORTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)

        if len(template_list) == 1:
            template_list = template_list[0]

        # ====distillation====
        if self.net.is_distill_training:
            with torch.no_grad():
                out_dict_teacher = self.net_teacher(template=template_list,
                                                    search=search_img, is_distill=True)
        # ==========================

        if 'periph_images' in data:  # 실험: 주변시 입력
            out_dict = self.net(template=template_list, search=search_img, periph=data['periph_images'][0],
                                pcoord=data['periph_coord'][0], is_distill=False)
        else:
            out_dict = self.net(template=template_list,
                                search=search_img, is_distill=False)

        if self.net.is_distill_training:
            feat_teacher = out_dict_teacher['backbone_feat']
            feat_student = out_dict['backbone_feat']
            distill_loss = torch.tensor([torch.nn.functional.mse_loss(feat_teacher[i], feat_student[i]) for i in range(feat_student.shape[0])]).cuda()
            out_dict['distill_loss'] = distill_loss

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
        # 실험: rollout-matched crop — 대상 중심이 crop 밖인 표본은 heatmap 전부 0, 박스 loss 제외
        vis = gt_dict['search_visible'][-1].float().to(gt_gaussian_maps.device) if 'search_visible' in gt_dict else None
        if vis is not None:
            gt_gaussian_maps = gt_gaussian_maps * vis.view(-1, 1, 1, 1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        if vis is not None and (vis < 1).any():
            keep = vis.repeat_interleave(num_queries) > 0
            if keep.any():
                giou_loss, iou, giou = self.objective['giou'](pred_boxes_vec[keep], gt_boxes_vec[keep])
                l1_loss = self.objective['l1'](pred_boxes_vec[keep], gt_boxes_vec[keep])
            else:
                giou_loss = l1_loss = pred_boxes_vec.sum() * 0.0
                iou = giou = torch.zeros(1, device=pred_boxes_vec.device)
        else:
            try:
                giou_loss, iou, giou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss, iou, giou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum

        sim_loss = pred_dict['sim_loss']

        if self.net.is_distill_training:
            distill_loss = pred_dict['distill_loss']
            tau_0 = 10
            rho = 10
            coef = self.loss_weight['distill_loss'] * (tau_0 + rho * ((1 - giou) - (1 - giou).mean()))
            distill_loss = coef * distill_loss
            distill_loss = distill_loss.mean()
            loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss + self.loss_weight['sim_loss'] * sim_loss + distill_loss
        else:
            loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss + self.loss_weight['sim_loss'] * sim_loss
        pstat = {}
        if 'periph_logits' in pred_dict:  # 실험: 주변시/배치 head 손실
            loss_p, pstat = self.periph_losses(pred_dict, gt_dict)
            loss = loss + loss_p
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            if vis is not None:
                status["Vis"] = vis.mean().item()
            status.update(pstat)
            return loss, status
        else:
            return loss

    def periph_losses(self, pred, gt):
        import torch.nn.functional as F
        P = self.cfg.MODEL.PERIPH
        dev = pred['periph_logits'].device
        coord = gt['periph_coord'][0].to(dev); cur = gt['periph_cur'][0].to(dev)
        nxt = gt['periph_next'][0].to(dev); nv = gt['periph_next_valid'][0].to(dev).float()
        G = int(round(pred['periph_logits'].shape[1] ** .5))
        # (d) 주변시 현재 위치: 셀 CE + 셀 내부 offset L1 (대상 중심이 프레임 밖이면 제외)
        inf = ((cur >= 0) & (cur < 1)).all(-1).float()
        cxy = (cur.clamp(0, 1 - 1e-6) * G)
        ci = cxy.floor().long(); idx = ci[:, 1] * G + ci[:, 0]
        ce = F.cross_entropy(pred['periph_logits'], idx, reduction='none')
        off = pred['periph_off'][torch.arange(idx.shape[0], device=dev), idx]
        l_off = (off - (cxy - ci.float())).abs().sum(-1)
        den = inf.sum().clamp(min=1)
        l_cls = (ce * inf).sum() / den; l_off = (l_off * inf).sum() / den
        # (b) 다음 프레임 중심 (crop 한 변 단위, ±8 clip) + log 크기
        if P.NEXT_MODE == 'residual':  # 다음 중심 - 카메라 보정된 현재 GT 중심 (라벨 있는 표본만)
            cam = gt['periph_cam'][0].to(dev)
            dxy_gt = ((nxt[:, :2] - cam[:, :2]) / coord[:, 2:4]).clamp(-2, 2)
            nv = nv * cam[:, 2]
        else:
            dxy_gt = ((nxt[:, :2] - coord[:, :2]) / coord[:, 2:4]).clamp(-8, 8)
        l_next = F.smooth_l1_loss(pred['place_dxy'], dxy_gt, beta=0.5, reduction='none').sum(-1)
        l_ls = F.smooth_l1_loss(pred['place_ls'], nxt[:, 2].clamp(-4, 4), beta=0.5, reduction='none')
        dn = nv.sum().clamp(min=1)
        l_next = (l_next * nv).sum() / dn; l_ls = (l_ls * nv).sum() / dn
        # (c) 존재: 대상 중심이 현재 crop 안
        vis = gt['search_visible'][-1].float().to(dev)
        l_pres = F.binary_cross_entropy_with_logits(pred['presence_logit'], vis)
        loss = P.CLS_W * l_cls + P.OFF_W * l_off + P.NEXT_W * l_next + P.LS_W * l_ls + P.PRES_W * l_pres
        with torch.no_grad():
            est, _ = self.net.module.periph_estimate(pred) if hasattr(self.net, 'module') else self.net.periph_estimate(pred)
            perr = ((est - cur).abs() / coord[:, 2:4]).max(-1).values      # crop 한 변 단위 오차
            hit = ((perr <= .5).float() * inf).sum() / den                 # 추정 중심에 crop 을 놓으면 대상 중심이 crop 안
            pacc = ((torch.sigmoid(pred['presence_logit']) > .5).float() == vis).float().mean()
        st = {"P/cls": l_cls.item(), "P/off": l_off.item(), "P/next": l_next.item(), "P/ls": l_ls.item(),
              "P/pres": l_pres.item(), "P/hit": hit.item(), "P/pacc": pacc.item()}
        return loss, st

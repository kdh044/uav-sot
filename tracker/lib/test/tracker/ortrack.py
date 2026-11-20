import math
import numpy as np

from lib.models.ortrack import build_ortrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.test.tracker.motion_prior import MotionPrior
from lib.test.tracker.ego_motion import rotated_crop, map_back_rotated
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class ORTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(ORTrack, self).__init__(params)
        network = build_ortrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu', weights_only=False)['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        # self.network = network.cpu()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.feat_template_sz = self.cfg.TEST.TEMPLATE_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        # self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cpu()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.is_distill = params.cfg.MODEL['IS_DISTILL']

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template
        self.z_init = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        self.pf_first_image = image
        # 실험: 검색 중심 사전 예측 (env SOT_MOTION=cmc|cv|cmc+cv, 미설정 시 원본 동작)
        mode = os.environ.get('SOT_MOTION')
        if mode == 'learned':  # 실험: 학습된 주변시 ego-motion (CMC 대체)
            from lib.test.tracker.motion_prior import LearnedMotion
            self.motion = LearnedMotion(os.environ['SOT_EGO_CKPT'])
        else:
            if mode == 'phase':  # 실험(2026-11-13): 위상 상관 경량 CMC
                from lib.test.tracker.motion_prior import PhaseCorrMotion
                self.motion = PhaseCorrMotion(work_width=int(os.environ.get('SOT_PC_W', '128')))
            elif mode == 'kf':  # 실험(2026-11-13): CMC 보정 좌표 등속 KF
                from lib.test.tracker.motion_prior import CMCKalman
                self.motion = CMCKalman(work_width=int(os.environ.get('SOT_CMC_W', '640')))
            else:
                self.motion = MotionPrior(mode, work_width=int(os.environ.get('SOT_CMC_W', '640'))) if mode else None
        if self.motion:
            self.motion.init(image, info['init_bbox'])
        # 실험: M2 회전 정렬 crop (env SOT_ROT=1, SOT_MOTION 에 cmc 필요)
        self.use_rot = bool(os.environ.get('SOT_ROT')) and self.motion is not None and self.motion.use_cmc
        self.rot_wh = (info['init_bbox'][2], info['init_bbox'][3])
        self.ego_log = [] if os.environ.get('SOT_LOG_DIR') else None
        # 실험: 검색 crop 기록 (env SOT_CROP_LOG=dir) — [frame, crop_cx, crop_cy, crop_w, crop_h, factor, score_max]
        self.crop_log = [] if os.environ.get('SOT_CROP_LOG') else None
        self.low_cnt = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        if os.environ.get('SOT_REACQ'):  # 실험(2026-11-13): 핵심판 + DAM4SAM 식 다시 찾기
            from lib.test.tracker.sm_track import track_reacq
            return track_reacq(self, image)
        if os.environ.get('SOT_SIMPLE'):  # 실험(2026-11-13): 핵심판 — 두 위치 배치 추적 + 크기 맞춤 최소 박스
            from lib.test.tracker.sm_track import track_simple
            return track_simple(self, image)
        if os.environ.get('SOT_SM_BEAM'):  # 실험(2026-11-13): 가설 트리 + 방해물 메모리
            from lib.test.tracker.sm_track import track_beam
            return track_beam(self, image)
        if os.environ.get('SOT_SM'):  # 실험(2026-11-13): 학습 없는 상태 흐름 추적 (lib/test/tracker/sm_track.py)
            from lib.test.tracker.sm_track import track_sm
            return track_sm(self, image)
        if hasattr(self.network, 'periph_estimate'):  # 실험: 주변시+중심시 모델
            return self._track_periph(image)
        H, W, _ = image.shape
        self.frame_id += 1
        # 실험: 작은 대상이면 검색 배율 축소 (env SOT_ADAPT_SF="thr_px:factor", 미설정 시 원본 동작)
        search_factor = self.params.search_factor
        adapt = os.environ.get('SOT_ADAPT_SF')
        if adapt:
            thr, small_sf = map(float, adapt.split(':'))
            if (self.state[2] * self.state[3]) ** 0.5 < thr:
                search_factor = small_sf
        if self.motion:
            self.state = self.motion.predict(image, self.state)
        search_center = (self.state[0] + self.state[2] / 2, self.state[1] + self.state[3] / 2)
        prev_wh, prev_rot_wh = (self.state[2], self.state[3]), self.rot_wh
        rot_now = self.use_rot
        if self.use_rot:
            angle = 0.0 if os.environ.get('SOT_ROT_ZERO') else self.motion.cum_angle * float(os.environ.get('SOT_ROT_SIGN', '1'))
            # 회전이 작으면 일반 crop 사용 (env SOT_ROT_MIN 도 단위)
            if abs(math.degrees(angle)) < float(os.environ.get('SOT_ROT_MIN', '0')):
                rot_now = False
        if rot_now:
            w_al, h_al = self.rot_wh
            box_al = [search_center[0] - w_al / 2, search_center[1] - h_al / 2, w_al, h_al]
            x_patch_arr, resize_factor, x_amask_arr, rot_M = rotated_crop(image, box_al, search_factor,
                                                                          self.params.search_size, angle)
        if not rot_now:
            x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, search_factor,
                                                                    output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        if self.crop_log is not None:
            self.crop_log.append([self.frame_id, self.state[0] + self.state[2] / 2, self.state[1] + self.state[3] / 2,
                                  self.state[2], self.state[3], search_factor, float('nan')])

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, is_distill=self.is_distill)

        # add hann windows
        pred_score_map = out_dict['score_map']
        if self.crop_log is not None:
            self.crop_log[-1][-1] = float(pred_score_map.max())
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        if rot_now:
            crop_px = (pred_boxes.mean(dim=0) * self.params.search_size).tolist()
            al, aabb = map_back_rotated(crop_px, rot_M, angle)
            box_mode = os.environ.get('SOT_ROT_BOX')
            if box_mode == 'keep':  # 위치만 회전 정렬 결과 사용, 크기는 직전 값 유지
                aabb = [al[0] - prev_wh[0] / 2, al[1] - prev_wh[1] / 2, prev_wh[0], prev_wh[1]]
            else:
                self.rot_wh = (al[2], al[3])
                if box_mode == 'aligned':  # 외접 사각형 대신 정렬 프레임 크기 그대로 출력
                    aabb = [al[0] - al[2] / 2, al[1] - al[3] / 2, al[2], al[3]]
            self.state = clip_box(aabb, H, W, margin=int(os.environ.get('SOT_FLOOR', '10')))
        else:
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=int(os.environ.get('SOT_FLOOR', '10')))
            if self.use_rot:
                self.rot_wh = (self.state[2], self.state[3])
        # 실험: M6 템플릿 갱신 (env SOT_TUPD="thr:interval:mode", mode=replace|mix) — 신뢰도 높을 때 최근 모습으로 템플릿 교체
        tupd = os.environ.get('SOT_TUPD')
        if tupd:
            thr, interval, mode = tupd.split(':')
            if self.frame_id % int(interval) == 0 and float(pred_score_map.max()) >= float(thr):
                z_arr, _, z_m = sample_target(image, self.state, self.params.template_factor,
                                              output_sz=self.params.template_size)
                newz = self.preprocessor.process(z_arr, z_m)
                if mode == 'mix':  # 첫 템플릿과 평균 (드리프트 완화)
                    newz.tensors = 0.5 * self.z_init.tensors + 0.5 * newz.tensors
                self.z_dict1 = newz
        # 실험: M5 저신뢰 유지 (env SOT_HOLD=thr) — 점수가 thr 미만이면 ViT 출력 대신 예측 위치(CMC)에 머무름
        hold = os.environ.get('SOT_HOLD')
        if hold and not self.use_rot and float(pred_score_map.max()) < float(hold):
            self.state = clip_box([search_center[0] - prev_wh[0] / 2, search_center[1] - prev_wh[1] / 2,
                                   prev_wh[0], prev_wh[1]], H, W, margin=10)
        # 실험: M4 놓침 시 재탐색 (env SOT_RESEARCH="thr:k:grid:accept", 예 0.35:2:1:0.5)
        research = os.environ.get('SOT_RESEARCH')
        if research and not self.use_rot:
            thr, k, grid, accept = research.split(':')
            thr, k, grid, accept = float(thr), int(k), int(grid), float(accept)
            cur = float(pred_score_map.max())
            self.low_cnt = self.low_cnt + 1 if cur < thr else 0
            if self.low_cnt >= k:
                self.state = self._research(image, search_center, prev_wh, search_factor, grid, accept, cur, H, W)
        # 실험: 프레임당 크기 변화 제한 (env SOT_SIZE_CLAMP=r, 예 1.2 → [1/1.2, 1.2] 배)
        clamp = os.environ.get('SOT_SIZE_CLAMP')
        if clamp:
            r = float(clamp)
            cx, cy = self.state[0] + self.state[2] / 2, self.state[1] + self.state[3] / 2
            nw = min(max(self.state[2], prev_wh[0] / r), prev_wh[0] * r)
            nh = min(max(self.state[3], prev_wh[1] / r), prev_wh[1] * r)
            self.state = [cx - nw / 2, cy - nh / 2, nw, nh]
            if self.use_rot:
                aw, ah = self.rot_wh
                self.rot_wh = (min(max(aw, prev_rot_wh[0] / r), prev_rot_wh[0] * r),
                               min(max(ah, prev_rot_wh[1] / r), prev_rot_wh[1] * r))
        if self.motion:
            self.motion.update(self.state)
        if self.ego_log is not None:
            oc = (self.state[0] + self.state[2] / 2, self.state[1] + self.state[3] / 2)
            self.ego_log.append([self.frame_id, search_center[0], search_center[1], oc[0], oc[1],
                                 float(pred_score_map.max()), float(self.motion is not None and self.motion.H is not None),
                                 float(self.motion.cum_angle) if self.motion else 0.0,
                                 self.state[2], self.state[3]])

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    # ---------------- 실험(2026-11-12): 주변시 + 중심시 추론 ----------------
    # env: SOT_PF_NEXT = cmc(기본: 직전 상태(+SOT_MOTION)) | head(배치 head 예측) | mix(둘 평균)
    #      SOT_PF_PRES = 존재 임계값(기본 0.5), SOT_PF_REACQ = none | next(기본, 주변시 추정 위치로 상태 이동) | same(같은 프레임 재-crop 후 재추론)
    #      SOT_PF_PCONF = 재획득에 필요한 주변시 최대확률(기본 0), SOT_PF_LOG = 프레임별 로그 저장 폴더
    def _pf_forward(self, image, state, periph, sf=None):
        H, W, _ = image.shape
        sf = sf or self.params.search_factor
        x_arr, rf, x_m = sample_target(image, state, sf, output_sz=self.params.search_size)
        search = self.preprocessor.process(x_arr, x_m)
        cx, cy = state[0] + state[2] / 2, state[1] + state[3] / 2
        side = math.ceil(math.sqrt(state[2] * state[3]) * sf)
        pc = torch.tensor([[cx / W, cy / H, side / W, side / H]], dtype=torch.float32, device='cuda')
        with torch.no_grad():
            out = self.network.forward(template=self.z_dict1.tensors, search=search.tensors, periph=periph, pcoord=pc)
            est, conf = self.network.periph_estimate(out) if periph is not None else (None, None)
        resp = self.output_window * out['score_map']
        ksel = int(os.environ.get('SOT_KF_SELECT', '0'))
        if ksel > 1 and hasattr(self.motion, 'd2'):  # 실험: 점수 지도 상위 후보 중 KF 예측과 맞는 것 선택 (SAMURAI식 방해물 거르기)
            r = resp[0, 0]; Sz = r.shape[-1]; flat = r.flatten().clone(); cands = []
            for _ in range(ksel):
                j = int(flat.argmax()); v = float(flat[j])
                if v <= 0: break
                cands.append((j, v)); y0, x0 = divmod(j, Sz)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y0 + dy, x0 + dx
                        if 0 <= yy < Sz and 0 <= xx < Sz: flat[yy * Sz + xx] = 0
            lam = float(os.environ.get('SOT_KF_LAMBDA', '9')); best = None
            cxs, cys = state[0] + state[2] / 2, state[1] + state[3] / 2
            for j, v in cands:
                oh = torch.zeros_like(resp).flatten(); oh[j] = 1; oh = oh.view_as(resp)
                b = (self.network.box_head.cal_bbox(oh, out['size_map'], out['offset_map']).view(-1, 4)[0] * self.params.search_size / rf).tolist()
                h0 = 0.5 * self.params.search_size / rf
                fb = [b[0] + cxs - h0 - b[2] / 2, b[1] + cys - h0 - b[3] / 2, b[2], b[3]]
                sc = v * math.exp(-0.5 * min(self.motion.d2(fb), 50.0) / lam)
                if best is None or sc > best[0]: best = (sc, j)
            if best is not None and len(cands) > 1:
                oh = torch.zeros_like(resp).flatten(); oh[best[1]] = 1; resp = oh.view_as(resp)
        self.last_vis = (x_arr, out['score_map'][0, 0].detach().cpu().numpy())  # 시각화용: 마지막 search crop + 점수 지도
        pb = self.network.box_head.cal_bbox(resp, out['size_map'], out['offset_map']).view(-1, 4)
        pb = (pb.mean(0) * self.params.search_size / rf).tolist()
        half = 0.5 * self.params.search_size / rf
        box = [pb[0] + cx - half - pb[2] / 2, pb[1] + cy - half - pb[3] / 2, pb[2], pb[3]]
        if periph is None:  # 국소만 (주변시 꺼짐): 존재=1, 재획득 없음
            return dict(p_local=1.0, est_in=True, box=box, pres=1.0, score=float(out['score_map'].max()), est=(cx, cy), conf=0.0,
                        nxt=(cx, cy), crop=(cx, cy, side), ls=0.0)
        d = out['place_dxy'][0].tolist()
        G = self.network.p_grid
        prob = out['periph_logits'][0].softmax(-1)
        ci = min(max(int(cx / W * G), 0), G - 1) + G * min(max(int(cy / H * G), 0), G - 1)
        ex, ey = float(est[0, 0]) * W, float(est[0, 1]) * H
        return dict(p_local=float(prob[ci]), est_in=bool(abs(ex - cx) <= side / 2 and abs(ey - cy) <= side / 2), box=box, pres=float(torch.sigmoid(out['presence_logit'][0])), score=float(out['score_map'].max()),
                    est=(float(est[0, 0]) * W, float(est[0, 1]) * H), conf=float(conf[0]),
                    nxt=(cx + d[0] * side, cy + d[1] * side), crop=(cx, cy, side), ls=float(out['place_ls'][0]))

    def _track_periph(self, image):
        H, W, _ = image.shape
        self.frame_id += 1
        E = os.environ
        if not hasattr(self, 'pf_cfg'):
            self.pf_cfg = dict(next=E.get('SOT_PF_NEXT', 'cmc'), tau=float(E.get('SOT_PF_PRES', '0.5')),
                               reacq=E.get('SOT_PF_REACQ', 'next'), pconf=float(E.get('SOT_PF_PCONF', '0')),
                               # gate: 주변시 최대확률 >= MARGIN x crop 중심 셀 확률, 추정이 crop 밖, 최근 카메라 이동 >= CAM (정규화)
                               margin=float(E.get('SOT_PF_GATE_MARGIN', '0')), cam=float(E.get('SOT_PF_GATE_CAM', '0')),
                               outside=bool(int(E.get('SOT_PF_GATE_OUT', '0'))),
                               # 축② crop 크기: FLOOR = 상태 박스 최소 변(px, 원본 10), SF_LOW = 존재확률<tau 다음 프레임 배율 곱,
                               # MINSIDE = crop 한 변 최소(px), LS = 1이면 place_ls(다음 크기 변화)로 상태 크기 갱신
                               floor=float(E.get('SOT_FLOOR', '10')), sf_low=float(E.get('SOT_PF_SF_LOW', '1')),
                               minside=float(E.get('SOT_PF_MINSIDE', '0')), ls=bool(int(E.get('SOT_PF_LS', '0'))),
                               # 트리거 "score:cam:d2": 국소 점수<score 또는 카메라 이동/crop 한 변>cam 또는 KF d2>d2 일 때만 주변시 실행 (빈 값 = 매 프레임)
                               trig=[float(v) for v in E['SOT_PF_TRIGGER'].split(':')] if E.get('SOT_PF_TRIGGER') else None)
            self.pf_trig = [0, 0]
            self.pf_log = [] if E.get('SOT_PF_LOG') else None
            mean = torch.tensor(self.cfg.DATA.MEAN).view(1, 3, 1, 1).cuda(); std = torch.tensor(self.cfg.DATA.STD).view(1, 3, 1, 1).cuda()
            self.pf_norm = (mean, std)
        if self.frame_id == 1:
            self.pf_next = None; self.pf_res = None; self.pf_cam_hist = []; self.pf_low = False
            if self.pf_cfg['cam'] > 0 and not (self.motion is not None and hasattr(self.motion, 'net')):
                from lib.test.tracker.motion_prior import LearnedMotion   # gate 전용 카메라 움직임 추정 (crop 은 옮기지 않음)
                self.pf_gate_ego = LearnedMotion(E['SOT_EGO_CKPT']); self.pf_gate_ego.init(self.pf_first_image, None)
        C = self.pf_cfg
        prev_wh = (self.state[2], self.state[3])
        if self.motion:
            self.state = self.motion.predict(image, self.state)
        if C['next'] == 'residual' and getattr(self, 'pf_res', None) is not None:
            # 분해형: 카메라 움직임 항(위 motion.predict) + 학습된 대상 움직임 잔차 (직전 프레임 crop 한 변 단위 → 픽셀)
            self.state = [self.state[0] + self.pf_res[0], self.state[1] + self.pf_res[1], self.state[2], self.state[3]]
        if C['next'] in ('head', 'mix') and self.pf_next is not None:
            cx, cy = self.pf_next
            if C['next'] == 'mix':
                cx = 0.5 * (cx + self.state[0] + self.state[2] / 2); cy = 0.5 * (cy + self.state[1] + self.state[3] / 2)
            self.state = [cx - self.state[2] / 2, cy - self.state[3] / 2, self.state[2], self.state[3]]
        sf = self.params.search_factor * (C['sf_low'] if self.pf_low else 1.0)
        if C['minside'] > 0:
            sf = max(sf, C['minside'] / max(math.sqrt(self.state[2] * self.state[3]), 1e-3))
        use_p = True
        if C['trig'] is not None:
            o = self._pf_forward(image, self.state, None, sf)
            st, ct, kt = C['trig']
            camr = getattr(self.motion, 'cam', 0.0) / max(o['crop'][2], 1.0)
            kd = self.motion.d2(o['box']) if hasattr(self.motion, 'd2') else 0.0
            use_p = o['score'] < st or camr > ct or kd > kt
            self.pf_trig[0] += 1; self.pf_trig[1] += int(use_p)
        R = self.network.p_res
        if use_p:
            p = cv2.resize(image, (R, R), interpolation=cv2.INTER_AREA)
            p = torch.from_numpy(p).cuda().permute(2, 0, 1)[None].float() / 255.
            p = (p - self.pf_norm[0]) / self.pf_norm[1]
            o = self._pf_forward(image, self.state, p, sf)
        else:
            p = None
        if self.crop_log is not None:
            self.crop_log.append([self.frame_id, o['crop'][0], o['crop'][1], self.state[2], self.state[3],
                                  self.params.search_factor, o['score']])
        reacq = 0
        new = o['box']
        cam = 0.0
        if C['cam'] > 0:
            ego = self.motion if (self.motion is not None and hasattr(self.motion, 'net')) else self.pf_gate_ego
            if ego is not self.motion:
                ego.predict(image, [W / 2, H / 2, 1, 1])
            Hn = ego.H
            if Hn is not None and np.isfinite(Hn).all():
                q = Hn @ np.array([.5, .5, 1.0]); cam = float(np.hypot(q[0] / q[2] - .5, q[1] / q[2] - .5))
            self.pf_cam_hist = (self.pf_cam_hist + [cam])[-10:]
        gate_ok = (o['conf'] >= C['margin'] * o['p_local']) and (not C['outside'] or not o['est_in']) and \
                  (C['cam'] <= 0 or max(self.pf_cam_hist) >= C['cam'])
        if o['pres'] < C['tau'] and C['reacq'] != 'none' and o['conf'] >= C['pconf'] and gate_ok:
            g = o['est']
            gbox = [g[0] - prev_wh[0] / 2, g[1] - prev_wh[1] / 2, prev_wh[0], prev_wh[1]]
            reacq = 1
            new = gbox
            if C['reacq'] == 'same':
                o2 = self._pf_forward(image, clip_box(gbox, H, W, margin=C['floor']), p, sf)
                if o2['pres'] >= C['tau']:
                    new, o, reacq = o2['box'], o2, 2
        if C['ls'] and reacq == 0:
            k = math.exp(max(min(o['ls'], 0.5), -0.5)) ** 0.5   # 다음 프레임 크기 변화의 절반만 반영
            new = [new[0] + new[2] * (1 - k) / 2, new[1] + new[3] * (1 - k) / 2, new[2] * k, new[3] * k]
        self.state = clip_box(new, H, W, margin=C['floor'])
        self.pf_low = o['pres'] < C['tau']
        tu = E.get('SOT_PF_TUPD')  # 실험(④ 메모리): "score:interval:mode" 존재·점수 모두 높을 때만 템플릿 갱신, 재획득 시 첫 템플릿으로 복귀
        if tu:
            thr, itv, mode = tu.split(':')
            if reacq:
                self.z_dict1 = self.z_init
            elif o['pres'] >= 0.9 and o['score'] >= float(thr) and self.frame_id % int(itv) == 0:
                z_arr, _, z_m = sample_target(image, self.state, self.params.template_factor, output_sz=self.params.template_size)
                newz = self.preprocessor.process(z_arr, z_m)
                if mode == 'mix':
                    newz.tensors = 0.5 * self.z_init.tensors + 0.5 * newz.tensors
                self.z_dict1 = newz
        self.pf_next = (min(max(o['nxt'][0], 0.0), W - 1.0), min(max(o['nxt'][1], 0.0), H - 1.0))  # 프레임 안으로 제한
        side = o['crop'][2]
        self.pf_res = ((o['nxt'][0] - o['crop'][0]), (o['nxt'][1] - o['crop'][1]))  # residual 모델: place_dxy*side (픽셀)
        if self.motion:
            self.motion.update(self.state)
        if self.pf_log is not None:
            self.pf_log.append([self.frame_id, o['pres'], o['score'], o['est'][0], o['est'][1], o['conf'], reacq,
                                o['crop'][0], o['crop'][1], o['crop'][2], o['p_local'], cam, int(use_p)])  # 마지막 열: 주변시 실행 여부
        return {"target_bbox": self.state}

    def _research(self, image, center, wh, search_factor, grid, accept, cur_score, H, W):
        """검색 중심 주변 (2*grid+1)^2 위치(한 칸 = 검색영역 한 변)를 배치로 탐색해 가장 높은 점수 후보로 이동."""
        side = math.sqrt(wh[0] * wh[1]) * search_factor
        patches, factors, boxes = [], [], []
        for dy in range(-grid, grid + 1):
            for dx in range(-grid, grid + 1):
                if dx == 0 and dy == 0:
                    continue
                c = (center[0] + dx * side, center[1] + dy * side)
                if not (0 <= c[0] < W and 0 <= c[1] < H):
                    continue
                b = [c[0] - wh[0] / 2, c[1] - wh[1] / 2, wh[0], wh[1]]
                p, rf, m = sample_target(image, b, search_factor, output_sz=self.params.search_size)
                patches.append(self.preprocessor.process(p, m).tensors)
                factors.append(rf); boxes.append(b)
        if not patches:
            return self.state
        x = torch.cat(patches, 0)
        z = self.z_dict1.tensors.expand(x.shape[0], -1, -1, -1)
        with torch.no_grad():
            out = self.network.forward(template=z, search=x, is_distill=self.is_distill)
        resp = self.output_window * out['score_map']
        scores = out['score_map'].flatten(1).max(1).values
        i = int(scores.argmax())
        if float(scores[i]) < max(accept, cur_score):
            return self.state
        self.low_cnt = 0
        pb = self.network.box_head.cal_bbox(resp[i:i + 1], out['size_map'][i:i + 1], out['offset_map'][i:i + 1]).view(-1, 4)
        pb = (pb.mean(0) * self.params.search_size / factors[i]).tolist()
        cx_prev, cy_prev = boxes[i][0] + 0.5 * boxes[i][2], boxes[i][1] + 0.5 * boxes[i][3]
        half = 0.5 * self.params.search_size / factors[i]
        cx, cy, w, h = pb[0] + cx_prev - half, pb[1] + cy_prev - half, pb[2], pb[3]
        return clip_box([cx - 0.5 * w, cy - 0.5 * h, w, h], H, W, margin=10)

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return ORTrack

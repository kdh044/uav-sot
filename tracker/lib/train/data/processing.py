import math
import os
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


def _fast_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz):
    """rollout crop 전용: prutils.jittered_center_crop 와 같은 좌표계(crop_sz, x1 반올림, pad 규칙)이지만
    거대한 패딩 배열을 만들지 않고 이미지 안 영역만 출력 해상도로 resize 해 붙임 (rollout crop 은 대상의 수십 배 크기 가능 → OOM 방지)."""
    crops, boxes, atts = [], [], []
    for im, a, g in zip(frames, box_extract, box_gt):
        x, y, w, h = a.tolist()
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
        x1 = round(x + 0.5 * w - crop_sz * 0.5); y1 = round(y + 0.5 * h - crop_sz * 0.5)
        x2, y2 = x1 + crop_sz, y1 + crop_sz
        H, W = im.shape[:2]
        xa, xb = max(x1, 0), min(x2, W - 1); ya, yb = max(y1, 0), min(y2, H - 1)   # 원본 pad 규칙과 동일 범위
        rf = output_sz / crop_sz
        out = np.zeros((output_sz, output_sz, 3), dtype=im.dtype); att = np.ones((output_sz, output_sz), dtype=np.bool_)
        if xb > xa and yb > ya:
            dx0, dx1 = int(round((xa - x1) * rf)), int(round((xb - x1) * rf))
            dy0, dy1 = int(round((ya - y1) * rf)), int(round((yb - y1) * rf))
            dx1, dy1 = min(dx1, output_sz), min(dy1, output_sz)
            if dx1 > dx0 and dy1 > dy0:
                out[dy0:dy1, dx0:dx1] = cv.resize(im[ya:yb, xa:xb], (dx1 - dx0, dy1 - dy0))  # 테스트 sample_target 과 같은 INTER_LINEAR
                att[dy0:dy1, dx0:dx1] = False
        crops.append(out); atts.append(att)
        crop_t = torch.Tensor([output_sz, output_sz])
        boxes.append(prutils.transform_image_to_crop(g, a, rf, crop_t, normalize=True))
    return crops, boxes, atts, None


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search':  transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, ego_aug=False, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings
        self.ego_aug = ego_aug
        # 실험: rollout-matched search crop (cfg.DATA.SEARCH.ROLLOUT_BANK/PROB/OUTSIDE, 미설정 시 원본 동작)
        # bank: 실제 연속추적(uavft_tiny+CMC, BioDrone train)에서 기록한 (dx, dy, ls)
        #   dx,dy = (GT중심 - crop중심) / crop 한 변,  ls = log(sqrt(GT w*h) / sqrt(crop기준 w*h))
        self.rollout_prob = 0.0
        self.rollout_outside = 'zero'
        bank = getattr(settings, 'rollout_bank', None) if settings is not None else None
        if bank:
            z = np.load(bank)
            self.rollout_bank = np.stack([z['dx'], z['dy'], z['ls']], 1).astype(np.float64)
            self.rollout_prob = float(getattr(settings, 'rollout_prob', 0.5))
            self.rollout_outside = getattr(settings, 'rollout_outside', 'zero')
            # 실험: 실제 rollout 재현 (BioDrone train 표본, 해당 seq/frame 의 실제 crop)
            self.replay_prob = float(getattr(settings, 'replay_prob', 0.0) or 0.0)
            if self.replay_prob > 0:
                self.replay_index = {}
                for i, (sq, fr) in enumerate(zip(z['seq'], z['frame'])):
                    self.replay_index.setdefault(str(sq), []).append((int(fr), i))
        # 실험(축②): search 배율 범위 (settings.sf_range=[lo,hi], 비면 원본)
        self.sf_range = list(getattr(settings, 'sf_range', []) or []) if settings is not None else []
        self.sf_base = search_area_factor['search']
        # 실험: 주변시 입력 (settings.periph_res > 0 일 때만)
        self.periph_res = int(getattr(settings, 'periph_res', 0) or 0) if settings is not None else 0
        self.need_next = self.periph_res > 0
        self.next_mode = getattr(settings, 'next_mode', 'abs') if settings is not None else 'abs'
        self.ego_dir = getattr(settings, 'ego_dir', None) if settings is not None else None
        self._ego_cache = {}
        if self.periph_res:
            self.p_mean = np.array(settings.data_mean, np.float32).reshape(1, 1, 3)
            self.p_std = np.array(settings.data_std, np.float32).reshape(1, 1, 3)

    def ego_H(self, seq):
        """BioDrone train 시퀀스의 연속 프레임 호모그래피 (정규화 좌표, i→i+1). 없으면 None."""
        if self.next_mode != 'residual' or not self.ego_dir:
            return None
        if seq not in self._ego_cache:
            f = os.path.join(self.ego_dir, f'bd_{seq}_H.npy')
            self._ego_cache[seq] = np.load(f) if os.path.exists(f) else None
        return self._ego_cache[seq]

    @staticmethod
    def _flip_state(transform, cls):
        if transform is None:
            return False
        for t in transform.transforms:
            if type(t) is cls:
                rp = t._rand_params
                return bool(rp[0]) if rp else False
        return False

    def _periph(self, data, frame, crop_box):
        """전체 프레임 → periph_images (3,R,R 정규화), periph_coord [cu,cv,sw,sh], 현재/다음 GT (전체 프레임 정규화).
        flip: joint flip(프레임 전체)은 이미 frame/anno 에 반영됨 → 다음 GT 에만 적용. crop flip 은 '장면 전체 좌우 반전'과 동치로 처리."""
        import lib.train.data.transforms as tfm
        H, W = frame.shape[:2]
        x, y, w, h = [float(v) for v in crop_box]
        side = math.ceil(math.sqrt(w * h) * self.search_area_factor['search'])
        cu, cv_, sw, sh = (x + w / 2) / W, (y + h / 2) / H, side / W, side / H
        g = data['search_anno'][0]
        gu, gv = float(g[0] + g[2] / 2) / W, float(g[1] + g[3] / 2) / H
        n = data['search_next_anno'].clone().float()
        if self._flip_state(self.transform['joint'], tfm.RandomHorizontalFlip):
            n[0] = W - n[0] - n[2]
        nu, nv = float(n[0] + n[2] / 2) / W, float(n[1] + n[3] / 2) / H
        cam = data.get('search_next_cam', None)   # 카메라 움직임으로 옮긴 현재 GT 중심 (원본 좌표, 정규화), 없으면 None
        if cam is not None:
            cu_, cv2_ = float(cam[0]), float(cam[1])
            if self._flip_state(self.transform['joint'], tfm.RandomHorizontalFlip):
                cu_ = 1 - cu_
            nxt_cam = [cu_, cv2_, 1.0]
        else:
            nxt_cam = [nu, nv, 0.0]
        ls_next = math.log(max(math.sqrt(max(float(n[2] * n[3]), 1e-6)), 1e-3) / max(math.sqrt(w * h), 1e-3))
        img = cv.resize(frame, (self.periph_res, self.periph_res), interpolation=cv.INTER_AREA).astype(np.float32) / 255.
        return img, [cu, cv_, sw, sh], [gu, gv], [nu, nv, ls_next], nxt_cam

    def _periph_finish(self, data, img, coord, cur, nxt, cam):
        import lib.train.data.transforms as tfm
        if self._flip_state(self.transform['search'], tfm.RandomHorizontalFlip_Norm):
            img = img[:, ::-1]
            coord[0] = 1 - coord[0]; cur[0] = 1 - cur[0]; nxt[0] = 1 - nxt[0]; cam[0] = 1 - cam[0]
        img = (img - self.p_mean) / self.p_std
        data['periph_images'] = [torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))]
        data['periph_coord'] = [torch.tensor(coord, dtype=torch.float32)]
        data['periph_cur'] = [torch.tensor(cur, dtype=torch.float32)]
        data['periph_next'] = [torch.tensor(nxt, dtype=torch.float32)]
        data['periph_next_valid'] = [torch.tensor(float(data['search_next_valid']))]
        data['periph_cam'] = [torch.tensor(cam, dtype=torch.float32)]
        del data['search_next_anno'], data['search_next_valid']
        if 'search_next_cam' in data:
            del data['search_next_cam']

    @staticmethod
    def _ego_motion_aug(crop, box, att):
        """UAV 기체 흔들림 모사 (검색 crop 전용): 평면 회전 ±30° (p=0.5), 방향성 모션블러 3~9 (p=0.3).
        box: crop 기준 정규화 (x,y,w,h). att: 1=패딩."""
        S = crop.shape[0]
        if np.random.rand() < 0.5:
            ang = np.random.uniform(-30, 30)
            M = cv.getRotationMatrix2D(((S - 1) / 2, (S - 1) / 2), ang, 1.0)
            crop = cv.warpAffine(crop, M, (S, S), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
            att_np = att.numpy().astype(np.uint8) if torch.is_tensor(att) else att.astype(np.uint8)
            att_np = cv.warpAffine(att_np, M, (S, S), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue=1)
            att = torch.from_numpy(att_np.astype(bool)) if torch.is_tensor(att) else att_np.astype(att.dtype)
            x, y, w, h = (box * S).tolist()
            pts = np.array([[x, y, 1], [x + w, y, 1], [x, y + h, 1], [x + w, y + h, 1]]) @ M.T
            x0, y0 = pts.min(0); x1, y1 = pts.max(0)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, S), min(y1, S)
            box = torch.tensor([x0, y0, x1 - x0, y1 - y0], dtype=box.dtype) / S
        if np.random.rand() < 0.3:
            k = int(np.random.choice([3, 5, 7, 9]))
            ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
            R = cv.getRotationMatrix2D(((k - 1) / 2, (k - 1) / 2), np.random.uniform(0, 180), 1.0)
            ker = cv.warpAffine(ker, R, (k, k)); ker /= max(ker.sum(), 1e-6)
            crop = cv.filter2D(crop, -1, ker)
        return crop, box, att

    def _get_rollout_box(self, box, idx=None):
        """rollout bank 에서 (dx,dy,ls) 하나를 뽑아 crop 기준 박스(정사각) 생성. box: GT (x,y,w,h). idx: 재현할 bank 항목."""
        dx, dy, ls = self.rollout_bank[np.random.randint(len(self.rollout_bank)) if idx is None else idx]
        s = float(torch.sqrt(box[2] * box[3])) * math.exp(-ls)          # crop 기준 박스 한 변 (sqrt(w*h))
        side = s * self.search_area_factor['search']
        c = box[0:2] + 0.5 * box[2:4] - torch.tensor([dx, dy], dtype=box.dtype) * side
        return torch.cat((c - 0.5 * s, torch.tensor([s, s], dtype=box.dtype)), dim=0)

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

        if self.sf_range:  # 표본마다 search 배율 (log-uniform)
            lo, hi = self.sf_range
            self.search_area_factor['search'] = float(math.exp(np.random.uniform(math.log(lo), math.log(hi))))
        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            replay = s == 'search' and data.get('replay_idx', None) is not None
            use_roll = replay or (s == 'search' and self.rollout_prob > 0 and np.random.rand() < self.rollout_prob)
            if replay:
                jittered_anno = [self._get_rollout_box(a, idx=data['replay_idx']) for a in data[s + '_anno']]
            elif use_roll:
                jittered_anno = [self._get_rollout_box(a) for a in data[s + '_anno']]
            else:
                jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            if use_roll:
                crops, boxes, att_mask, mask_crops = _fast_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                                       self.search_area_factor[s], self.output_sz[s])
                # 마스크(copy-paste 증강용)는 이 설정에서 전부 0 → 출력 크기 0 텐서로 대체
                mask_crops = [torch.zeros((self.output_sz[s], self.output_sz[s])) for _ in crops]
            else:
              crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'], self.search_area_factor[s],
                                                                              self.output_sz[s], masks=data[s + '_masks'])
            if s == 'search':
                # 대상 중심이 crop 안이면 visible=1 (원본 라벨), 밖이면 0 (cls 전부 0, 박스 loss 제외)
                vis = [float(0 <= float(b[0] + b[2] / 2) <= 1 and 0 <= float(b[1] + b[3] / 2) <= 1) for b in boxes]
                if self.rollout_outside == 'reject' and min(vis) < 1:
                    data['valid'] = False
                    return data
                data['search_visible'] = [torch.tensor(v) for v in vis]
            if s == 'search' and self.ego_aug:
                out = [self._ego_motion_aug(c, b, a) for c, b, a in zip(crops, boxes, att_mask)]
                crops, boxes, att_mask = [o[0] for o in out], [o[1] for o in out], [o[2] for o in out]
                if any(b[2] <= 0 or b[3] <= 0 for b in boxes):
                    data['valid'] = False
                    return data
            if s == 'search' and self.periph_res:
                p_args = self._periph(data, data['search_images'][0], jittered_anno[0])
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)
            if s == 'search' and self.periph_res:
                self._periph_finish(data, *p_args)

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        data['valid'] = True
        if 'replay_idx' in data:
            del data['replay_idx']
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data

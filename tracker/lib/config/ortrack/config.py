from easydict import EasyDict as edict
import yaml

"""
Add default config for ORTrack.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
cfg.MODEL.IS_DISTILL = False
cfg.MODEL.EXTRA_MERGER = False

cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'

cfg.MODEL.BACKBONE.CE_LOC = []
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256

# 실험(2026-11-12): 주변시(전체 프레임 저해상도 토큰) + 중심시(search crop). ENABLE=False 이면 원본과 동일
cfg.MODEL.PERIPH = edict()
cfg.MODEL.PERIPH.ENABLE = False
cfg.MODEL.PERIPH.RES = 256      # 전체 프레임을 RES x RES 로 축소 → patch_embed(16) → POOL x POOL 평균 → (RES/16/POOL)^2 토큰
cfg.MODEL.PERIPH.POOL = 2
cfg.MODEL.PERIPH.CLS_W = 0.5    # 주변시 현재 위치 heatmap CE
cfg.MODEL.PERIPH.OFF_W = 1.0    # 주변시 셀 내부 offset L1
cfg.MODEL.PERIPH.NEXT_W = 1.0   # 다음 프레임 중심 (crop 한 변 단위) smooth L1
cfg.MODEL.PERIPH.LS_W = 0.5     # 다음 프레임 log 크기
cfg.MODEL.PERIPH.PRES_W = 1.0   # 현재 crop 안 대상 존재 BCE
cfg.MODEL.PERIPH.LR_MULT = 10.0 # 새 모듈(periph_*) 학습률 배수 (기준 TRAIN.LR)
cfg.MODEL.PERIPH.NEXT_MODE = 'abs'  # 배치 head 목표: abs(다음 중심 - crop 중심) | residual(다음 중심 - 카메라 움직임으로 옮긴 현재 중심; BioDrone train 만 라벨)
cfg.MODEL.PERIPH.EGO_DIR = '/home/airlab/user/kdh/SOT/analysis/peripheral_20261112/egomotion/data'
cfg.MODEL.PERIPH.SIDE = False      # True: 원본 ORTrack 동결 + 주변시 별도 가지 (SidePeriphORTrack)
cfg.MODEL.PERIPH.SIDE_BLOCKS = 2   # 주변시 가지에서 학습하는 마지막 block 복사본 수
cfg.MODEL.PERIPH.MERGE = 'avg'  # POOL x POOL patch 병합: avg | linear (concat→Linear, 평균으로 초기화)

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False

cfg.TRAIN.CE_START_EPOCH = 20  # candidate elimination start epoch
cfg.TRAIN.CE_WARM_EPOCH = 80  # candidate elimination warm up epoch
cfg.TRAIN.DROP_PATH_RATE = 0.1  # drop path rate for ViT backbone

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1
cfg.DATA.SEARCH.EGO_AUG = False  # 검색 crop 회전·모션블러 증강 (UAV ego-motion 모사)
cfg.DATA.SEARCH.ROLLOUT_BANK = ''   # rollout-matched crop: (dx,dy,ls) npz 경로, '' 이면 원본 jitter
cfg.DATA.SEARCH.FACTOR_RANGE = []  # 축②: [lo, hi] 이면 search 배율을 표본마다 log-uniform 으로 뽑음 (빈 값 = 원본 고정 배율)
cfg.DATA.SEARCH.ROLLOUT_PROB = 0.0  # search crop 을 bank 에서 뽑을 확률
cfg.DATA.SEARCH.ROLLOUT_OUTSIDE = 'zero'  # 대상 중심이 crop 밖: zero(cls 0, 박스 loss 제외) | reject(재샘플)
cfg.DATA.SEARCH.REPLAY_PROB = 0.0   # 실험: BioDrone train 표본을 확률 p 로 실제 rollout 기록(bank 의 seq,frame,dx,dy,ls) 그대로 재현 (distractor 포함 crop)
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)

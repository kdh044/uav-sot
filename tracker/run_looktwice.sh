#!/bin/bash
# Look Twice + 화면 밖 추적 (공식 ORTrack-DeiT 가중치 그대로, 학습 없음)
# usage: bash run_looktwice.sh <dataset_name> [runid]
#   dataset_name: uav123 | dtb70 | uavdt | visdrone2018 | biodrone | ...
cd "$(dirname "$0")/tracking"
export NVIDIA_TF32_OVERRIDE=0          # 평가 재현성 (TF32 켜면 점수가 1~2점 흔들림)
export SOT_SIMPLE=1 SOT_MOTION=cmc     # Look Twice: 직전 위치 + CMC 위치를 배치 2로 탐색, 점수 높은 쪽
export SOT_OOV=1 SOT_OOV_G=1.0         # 화면 밖 추적: 표적이 화면 밖이면 위치만 기억하고 대기
python test.py ortrack deit_tiny_patch16_224 --dataset_name "$1" --runid "${2:-1}" --threads 8 --num_gpus 1

#!/bin/bash
# 드론 미세조정 모델: 표준 3종 (UAV123 은 완료)
cd /home/airlab/user/kdh/SOT/ORTrack/tracking
export PYTHONPATH=../../pydeps NVIDIA_TF32_OVERRIDE=0
for ds in dtb70 uavdt visdrone2018; do
  python test.py ortrack deit_tiny_patch16_224_uavft --dataset_name $ds --threads 8 --num_gpus 2 2>&1 | grep -E "Done|Error" | tail -1
done
echo ALLDONE

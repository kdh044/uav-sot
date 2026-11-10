#!/bin/bash
# OSTrack-256 teacher: biodrone val + uav123 only (시간 제한으로 범위 축소), TF32 off
export PYTHONPATH=/home/airlab/user/kdh/SOT/pydeps NVIDIA_TF32_OVERRIDE=0
cd /home/airlab/user/kdh/SOT/OSTrack/tracking
for ds in biodrone uav123; do
  echo "=== vitb_256 $ds $(date +%T)"
  python test.py ostrack vitb_256_mae_ce_32x4_ep300 --dataset_name $ds --threads 4 --num_gpus 2 2>&1 | grep -E "Done|rror" | tail -1
done
echo ALLDONE

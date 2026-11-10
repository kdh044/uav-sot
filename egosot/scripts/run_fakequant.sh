#!/bin/bash
# ORTrack-D INT8 PTQ 모의: 변형 4종 x (uav123, biodrone val)
export NVIDIA_TF32_OVERRIDE=0 PYTHONPATH=/home/airlab/user/kdh/SOT/pydeps
unset SOT_ADAPT_SF SOT_MOTION
cd /home/airlab/user/kdh/SOT
for v in w8 w8a8_pct; do for ds in uav123 biodrone; do
  echo "=== $v $ds $(date +%T)"
  python tools/fakequant_eval.py run --variant $v --dataset $ds --workers 4 --out analysis/fq/${v}_$ds 2>&1 | grep -E "^done|Error" | tail -1
done; done
echo ALLDONE

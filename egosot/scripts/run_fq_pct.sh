#!/bin/bash
# 축소판: W8 BioDrone 종료 대기 후 W8A8-percentile (uav123, biodrone)만
export NVIDIA_TF32_OVERRIDE=0 PYTHONPATH=/home/airlab/user/kdh/SOT/pydeps
unset SOT_ADAPT_SF SOT_MOTION
cd /home/airlab/user/kdh/SOT
while pgrep -f "fakequant_eval.py run --variant w8 " >/dev/null; do sleep 5; done
echo "=== w8 biodrone finished $(date +%T)"
for ds in uav123 biodrone; do
  echo "=== w8a8_pct $ds $(date +%T)"
  python tools/fakequant_eval.py run --variant w8a8_pct --dataset $ds --workers 4 --out analysis/fq/w8a8_pct_$ds 2>&1 | grep -E "^done|Error" | tail -1
done
echo ALLDONE

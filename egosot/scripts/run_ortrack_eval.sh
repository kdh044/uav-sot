#!/bin/bash
cd /home/airlab/user/kdh/SOT/ORTrack/tracking
export PYTHONPATH=../../pydeps
export NVIDIA_TF32_OVERRIDE=0
for cfg in deit_tiny_patch16_224 deit_tiny_distilled_patch16_224; do
  for ds in dtb70 uavdt visdrone2018 uav123; do
    echo "=== $cfg $ds $(date)"
    python test.py ortrack $cfg --dataset_name $ds --threads 8 --num_gpus 2 2>&1 | grep -E "Done|Error|error" | tail -3
  done
done
echo ALLDONE

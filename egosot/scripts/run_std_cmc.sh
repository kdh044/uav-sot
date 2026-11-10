#!/bin/bash
# 표준 4종에서 CMC 손해 여부: ORTrack 원본+CMC, FT+CMC
cd /home/airlab/user/kdh/SOT/ORTrack/tracking
export PYTHONPATH=../../pydeps NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1 SOT_MOTION=cmc
for c in deit_tiny_patch16_224 deit_tiny_patch16_224_uavft; do
  for ds in uav123 dtb70 uavdt visdrone2018; do
    python test.py ortrack $c --dataset_name $ds --runid 801 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/stdcmc_${c}_$ds.log 2>&1
  done
done
echo ALLDONE

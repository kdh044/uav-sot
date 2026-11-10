#!/bin/bash
cd /home/airlab/user/kdh/SOT/ORTrack/tracking
export PYTHONPATH=../../pydeps NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1 SOT_MOTION=cmc
export SOT_BIODRONE=/home/airlab/user/kdh/datasets/sot/biodrone_tune
for t in 0.2:701 0.3:702 0.4:703; do
  SOT_HOLD=${t%%:*} python test.py ortrack deit_tiny_patch16_224_uavft --dataset_name biodrone --runid ${t##*:} --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/hold_${t##*:}.log 2>&1 &
done
wait
echo ALLDONE

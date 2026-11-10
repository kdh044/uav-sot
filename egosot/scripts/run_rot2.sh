#!/bin/bash
cd /home/airlab/user/kdh/SOT/ORTrack/tracking
export PYTHONPATH=../../pydeps NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1 SOT_MOTION=cmc SOT_ROT=1 SOT_ROT_BOX=keep
export SOT_BIODRONE=/home/airlab/user/kdh/datasets/sot/biodrone_tune
for t in 0:721 15:722 30:723; do
  SOT_ROT_MIN=${t%%:*} python test.py ortrack deit_tiny_patch16_224_uavft --dataset_name biodrone --runid ${t##*:} --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/rot2_${t##*:}.log 2>&1 &
done
wait
echo ALLDONE

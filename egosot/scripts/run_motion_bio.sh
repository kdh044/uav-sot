#!/bin/bash
# BioDrone val: 등속(cv) / CMC+등속 모션 사전예측, TF32 off
cd /home/airlab/user/kdh/SOT/ORTrack/tracking
export PYTHONPATH=../../pydeps NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1
SOT_MOTION=cv python test.py ortrack deit_tiny_patch16_224 --dataset_name biodrone --runid 102 --threads 6 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/mot_cv.log 2>&1 &
SOT_MOTION=cmc+cv python test.py ortrack deit_tiny_patch16_224 --dataset_name biodrone --runid 103 --threads 6 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/mot_cmccv.log 2>&1
wait

#!/bin/bash
# 미세조정 모델 2종 x (CMC 없음/있음) BioDrone 튜닝 분할 평가
cd /home/airlab/user/kdh/SOT/ORTrack/tracking
export PYTHONPATH=../../pydeps NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1
export SOT_BIODRONE=/home/airlab/user/kdh/datasets/sot/biodrone_tune
run(){ cfg=$1; id=$2; shift 2; env "$@" python test.py ortrack $cfg --dataset_name biodrone --runid $id --threads 6 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/ft_tune_${cfg}_$id.log 2>&1; }
run deit_tiny_patch16_224_uavft 400 DUMMY=1 &
run deit_tiny_patch16_224_uavft 401 SOT_MOTION=cmc &
run deit_tiny_patch16_224_uavft_ego 400 DUMMY=1 &
run deit_tiny_patch16_224_uavft_ego 401 SOT_MOTION=cmc
wait
echo ALLDONE

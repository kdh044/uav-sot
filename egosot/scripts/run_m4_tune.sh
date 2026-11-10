#!/bin/bash
# M4 재탐색: FT(드론 미세조정) + CMC 위에서 파라미터 sweep, BioDrone 튜닝 분할
cd /home/airlab/user/kdh/SOT/ORTrack/tracking
export PYTHONPATH=../../pydeps NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1
export SOT_BIODRONE=/home/airlab/user/kdh/datasets/sot/biodrone_tune SOT_MOTION=cmc
run(){ id=$1; shift; env "$@" python test.py ortrack deit_tiny_patch16_224_uavft --dataset_name biodrone --runid $id --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/m4_$id.log 2>&1; }
run 410 SOT_RESEARCH=0.35:2:1:0.5 &
run 411 SOT_RESEARCH=0.30:3:1:0.45 &
run 412 SOT_RESEARCH=0.40:2:1:0.55
wait
echo ALLDONE

#!/bin/bash
# BioDrone train 30개(튜닝 분할)에서 base / CMC+CV / CMC+CV+ROT, 내부 로그 저장
cd /home/airlab/user/kdh/SOT/ORTrack/tracking
export PYTHONPATH=../../pydeps NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1
export SOT_BIODRONE=/home/airlab/user/kdh/datasets/sot/biodrone_tune
L=/home/airlab/user/kdh/SOT/analysis/egolog
run(){ id=$1; shift; env "$@" SOT_LOG_DIR=$L/tune_$id python test.py ortrack deit_tiny_patch16_224 --dataset_name biodrone --runid $id --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/ego_tune_$id.log 2>&1; }
run 301 SOT_MOTION=cmc &   # cmc 만 켜고 로그 (base 비교용은 아래)
run 302 SOT_MOTION=cmc+cv
run 303 SOT_MOTION=cmc+cv SOT_ROT=1
wait
run 300 DUMMY=1            # base (모션 없음)
echo ALLDONE

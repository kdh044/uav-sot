#!/bin/bash
cd /home/airlab/user/kdh/SOT/ORTrack/tracking
export PYTHONPATH=../../pydeps NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1
export SOT_BIODRONE=/home/airlab/user/kdh/datasets/sot/biodrone_tune
L=/home/airlab/user/kdh/SOT/analysis/egolog
run(){ id=$1; shift; env "$@" SOT_LOG_DIR=$L/tune_$id python test.py ortrack deit_tiny_patch16_224 --dataset_name biodrone --runid $id --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/ego_tune_$id.log 2>&1; }
run 304 SOT_MOTION=cmc SOT_ROT=1 &
run 305 SOT_MOTION=cmc SOT_ROT=1 SOT_ROT_BOX=aligned
wait
echo ALLDONE

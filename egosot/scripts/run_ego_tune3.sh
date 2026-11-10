#!/bin/bash
cd /home/airlab/user/kdh/SOT/ORTrack/tracking
export PYTHONPATH=../../pydeps NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1
export SOT_BIODRONE=/home/airlab/user/kdh/datasets/sot/biodrone_tune
run(){ id=$1; shift; env "$@" python test.py ortrack deit_tiny_patch16_224 --dataset_name biodrone --runid $id --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/ego_tune_$id.log 2>&1; }
run 310 SOT_MOTION=cmc SOT_SIZE_CLAMP=1.2 &
run 311 SOT_MOTION=cmc SOT_ROT=1 SOT_SIZE_CLAMP=1.2 &
run 312 SOT_MOTION=cmc SOT_ROT=1 SOT_ROT_BOX=aligned SOT_SIZE_CLAMP=1.2
wait
run 313 SOT_SIZE_CLAMP=1.2
echo ALLDONE

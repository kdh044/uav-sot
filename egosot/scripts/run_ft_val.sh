#!/bin/bash
cd /home/airlab/user/kdh/SOT/ORTrack/tracking
export PYTHONPATH=../../pydeps NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1
run(){ cfg=$1; ds=$2; id=$3; shift 3; env "$@" python test.py ortrack $cfg --dataset_name $ds --runid $id --threads 6 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/ft_val_${cfg}_${ds}_$id.log 2>&1; }
run deit_tiny_patch16_224_uavft biodrone 501 SOT_MOTION=cmc &
run deit_tiny_patch16_224_uavft_ego biodrone 501 SOT_MOTION=cmc &
run deit_tiny_patch16_224_uavft uav123 500 DUMMY=1 &
run deit_tiny_patch16_224_uavft_ego uav123 500 DUMMY=1
wait
echo ALLDONE

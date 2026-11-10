#!/bin/bash
# UAV123@10fps (프레임 간 움직임 3배): 원본 / +CMC / 최종(B)+CMC
cd /home/airlab/user/kdh/SOT/ORTrack/tracking
export PYTHONPATH=../../pydeps NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1
python test.py ortrack deit_tiny_patch16_224 --dataset_name uav123_10fps --runid 980 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/u10_980.log 2>&1
SOT_MOTION=cmc python test.py ortrack deit_tiny_patch16_224 --dataset_name uav123_10fps --runid 981 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/u10_981.log 2>&1
SOT_MOTION=cmc python test.py ortrack deit_tiny_patch16_224_uavft_tiny --dataset_name uav123_10fps --runid 981 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/u10_b981.log 2>&1
echo ALLDONE

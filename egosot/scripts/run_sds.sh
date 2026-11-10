#!/bin/bash
# 해상(SeaDronesSee val 70 클립): 원본 / +CMC / 최종(B, +CMC)
cd /home/airlab/user/kdh/SOT/ORTrack/tracking
export PYTHONPATH=../../pydeps NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1
python test.py ortrack deit_tiny_patch16_224 --dataset_name seadronessee --runid 970 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/sds_970.log 2>&1
SOT_MOTION=cmc python test.py ortrack deit_tiny_patch16_224 --dataset_name seadronessee --runid 971 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/sds_971.log 2>&1
SOT_MOTION=cmc python test.py ortrack deit_tiny_patch16_224_uavft_tiny --dataset_name seadronessee --runid 971 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/sds_b971.log 2>&1
echo ALLDONE

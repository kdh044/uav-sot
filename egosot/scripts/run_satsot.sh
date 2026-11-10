#!/bin/bash
# 소형 대상(위성) SatSOT: 원본 / +CMC / FT / FT+CMC
cd /home/airlab/user/kdh/SOT/ORTrack/tracking
export PYTHONPATH=../../pydeps NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1
python test.py ortrack deit_tiny_patch16_224 --dataset_name satsot --runid 900 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/sat_900.log 2>&1
SOT_MOTION=cmc python test.py ortrack deit_tiny_patch16_224 --dataset_name satsot --runid 901 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/sat_901.log 2>&1
python test.py ortrack deit_tiny_patch16_224_uavft --dataset_name satsot --runid 900 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/sat_ft900.log 2>&1
SOT_MOTION=cmc python test.py ortrack deit_tiny_patch16_224_uavft --dataset_name satsot --runid 901 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/sat_ft901.log 2>&1
echo ALLDONE

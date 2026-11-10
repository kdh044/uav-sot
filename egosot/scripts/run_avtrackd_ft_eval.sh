#!/bin/bash
# AVTrack-D 드론 FT 모델: BioDrone val (CMC 유/무), UAV123 (CMC 무)
cd /home/airlab/user/kdh/SOT/AVTrack/tracking
export PYTHONPATH=/home/airlab/user/kdh/SOT/pydeps NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1
C=deit_tiny_distilled_patch16_224_uavft
python test.py avtrack $C --dataset_name biodrone --runid 88 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/avdft_bio.log 2>&1 &
SOT_MOTION=cmc python test.py avtrack $C --dataset_name biodrone --runid 89 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/avdft_bio_cmc.log 2>&1 &
python test.py avtrack $C --dataset_name uav123 --runid 88 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/avdft_uav123.log 2>&1
wait
echo ALLDONE

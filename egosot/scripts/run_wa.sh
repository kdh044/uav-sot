#!/bin/bash
# 가중치 평균(원본↔드론 FT) 모델: BioDrone tune (CMC) + 표준 4종
cd /home/airlab/user/kdh/SOT/ORTrack/tracking
export PYTHONPATH=../../pydeps NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1
for a in 5 7; do
  c=deit_tiny_patch16_224_uavft_wa$a
  SOT_BIODRONE=/home/airlab/user/kdh/datasets/sot/biodrone_tune SOT_MOTION=cmc python test.py ortrack $c --dataset_name biodrone --runid 430 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/wa${a}_tune.log 2>&1
  for ds in uav123 dtb70 uavdt visdrone2018; do
    python test.py ortrack $c --dataset_name $ds --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/wa${a}_$ds.log 2>&1
  done
done
echo ALLDONE

#!/bin/bash
# 최종 모델 B (드론 FT + LaTOT): 표준 4종 + CMC, BioDrone val CMC 없이 (ablation)
cd /home/airlab/user/kdh/SOT/ORTrack/tracking
export PYTHONPATH=../../pydeps NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1
C=deit_tiny_patch16_224_uavft_tiny
python test.py ortrack $C --dataset_name biodrone --runid 960 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/finalb_val_nocmc.log 2>&1
for ds in uav123 dtb70 uavdt visdrone2018; do
  SOT_MOTION=cmc python test.py ortrack $C --dataset_name $ds --runid 961 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/finalb_$ds.log 2>&1
done
echo ALLDONE

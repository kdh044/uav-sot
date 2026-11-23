#!/bin/bash
# 원본 OSTrack-256 CE (공식 가중치) 대표 세트. runid 51xx (xx=order index). 인자: 데이터 이름들
cd /home/airlab/user/kdh/SOT/OSTrack; O=/home/airlab/user/kdh/SOT/analysis/subset_20261113
order=(vetra birdsai biodrone uav123 dtb70 uavdt visdrone2018 sv248s satsot viso)
for k in "$@"; do for i in "${!order[@]}"; do [ "${order[$i]}" = "$k" ] && echo "51$(printf %02d $i) $k"; done; done | \
xargs -P 2 -L 1 bash -c 'id=$0; k=$1; g=$(( id % 2 )); env SOT_BIODRONE='"$O"'/$k OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/home/airlab/user/kdh/SOT/pydeps NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=$g python tracking/test.py ostrack vitb_256_mae_ce_32x4_ep300 --dataset_name biodrone --runid $id --threads 4 --num_gpus 1 > '"$O"'/run_$id.log 2>&1; echo done $id $k'

#!/bin/bash
# 검색 배율 5 로 드론 미세조정 → BioDrone tune/val (CMC 유무)
cd /home/airlab/user/kdh/SOT/ORTrack
export PYTHONPATH=/home/airlab/user/kdh/SOT/pydeps
C=deit_tiny_patch16_224_uavft_sf5
CK=output/checkpoints/train/ortrack/$C
CUDA_VISIBLE_DEVICES=0 python tracking/train.py --script ortrack --config $C --save_dir ./output --mode single > /home/airlab/user/kdh/SOT/logs/ft_sf5_train.log 2>&1 &
TP=$!
while kill -0 $TP 2>/dev/null; do   # 중간 체크포인트 정리 (마지막 것만 유지)
  ls $CK/ORTrack_ep00*.pth.tar 2>/dev/null | head -n -1 | xargs -r rm -f; sleep 60
done
ls $CK/ORTrack_ep0020.pth.tar || { echo TRAIN FAILED; exit 1; }
cd tracking; export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1
SOT_BIODRONE=/home/airlab/user/kdh/datasets/sot/biodrone_tune SOT_MOTION=cmc python test.py ortrack $C --dataset_name biodrone --runid 601 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/ft_sf5_tune.log 2>&1
SOT_MOTION=cmc python test.py ortrack $C --dataset_name biodrone --runid 602 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/ft_sf5_val.log 2>&1
python test.py ortrack $C --dataset_name uav123 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/ft_sf5_uav123.log 2>&1
echo ALLDONE

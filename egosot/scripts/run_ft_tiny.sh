#!/bin/bash
# 드론 FT + LaTOT(소형) 혼합 미세조정 → BioDrone tune/val(CMC), SatSOT(CMC), UAV123
cd /home/airlab/user/kdh/SOT/ORTrack
export PYTHONPATH=/home/airlab/user/kdh/SOT/pydeps
C=deit_tiny_patch16_224_uavft_tiny
CK=output/checkpoints/train/ortrack/$C
T0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python tracking/train.py --script ortrack --config $C --save_dir ./output --mode single > /home/airlab/user/kdh/SOT/logs/ft_tiny_train.log 2>&1 &
TP=$!
while kill -0 $TP 2>/dev/null; do
  ls $CK/ORTrack_ep00*.pth.tar 2>/dev/null | head -n -1 | xargs -r rm -f; sleep 60
done
ls $CK/ORTrack_ep00*.pth.tar 2>/dev/null | head -n -1 | xargs -r rm -f
echo "TRAIN_SEC $(( $(date +%s) - T0 ))"
[ -f $CK/ORTrack_ep0020.pth.tar ] || { echo TRAIN FAILED; exit 1; }
cd tracking; export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1
SOT_BIODRONE=/home/airlab/user/kdh/datasets/sot/biodrone_tune SOT_MOTION=cmc python test.py ortrack $C --dataset_name biodrone --runid 950 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/ft_tiny_tune.log 2>&1
SOT_MOTION=cmc python test.py ortrack $C --dataset_name biodrone --runid 951 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/ft_tiny_val.log 2>&1
SOT_MOTION=cmc python test.py ortrack $C --dataset_name satsot --runid 951 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/ft_tiny_sat.log 2>&1
python test.py ortrack $C --dataset_name uav123 --runid 952 --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/ft_tiny_uav123.log 2>&1
echo ALLDONE

#!/bin/bash
# AVTrack-DeiT(비증류) 드론 FT: ORTrack/AVTrack-D 와 같은 레시피, 이후 BioDrone val(CMC 유/무) + UAV123 평가
R=/home/airlab/user/kdh/SOT
cd $R/AVTrack
export PYTHONPATH=$R/pydeps
CK=$R/AVTrack/output/checkpoints/train/avtrack/deit_tiny_patch16_224_uavft
CUDA_VISIBLE_DEVICES=0 python tracking/train.py --script avtrack --config deit_tiny_patch16_224_uavft --save_dir ./output --mode single > $R/logs/avtrack_deit_ft_train.log 2>&1 &
TP=$!
while kill -0 $TP 2>/dev/null; do   # 학습 중 중간 체크포인트 정리 (최신 2개만 유지)
  ls -1 $CK/AVTrack_ep*.pth.tar 2>/dev/null | sort | head -n -2 | xargs -r rm -f
  sleep 60
done
ls -1 $CK/AVTrack_ep*.pth.tar 2>/dev/null | grep -v ep0020 | xargs -r rm -f
ls -la $CK
[ -f $CK/AVTrack_ep0020.pth.tar ] || { echo "TRAIN FAILED"; tail -30 $R/logs/avtrack_deit_ft_train.log; exit 1; }

cd $R/AVTrack/tracking
export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1
C=deit_tiny_patch16_224_uavft
python test.py avtrack $C --dataset_name biodrone --runid 90 --threads 8 --num_gpus 2 > $R/logs/avtrack_deit_ft_bio90.log 2>&1
SOT_MOTION=cmc python test.py avtrack $C --dataset_name biodrone --runid 91 --threads 8 --num_gpus 2 > $R/logs/avtrack_deit_ft_bio91.log 2>&1
python test.py avtrack $C --dataset_name uav123 --runid 90 --threads 8 --num_gpus 2 > $R/logs/avtrack_deit_ft_uav90.log 2>&1
echo ALLDONE

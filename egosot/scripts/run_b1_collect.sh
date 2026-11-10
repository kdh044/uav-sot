#!/bin/bash
# B1 특징 수집: BioDrone train(2 shard) + VisDrone-SOT train + BioDrone val, GPU1, 프로세스 4개
unset SOT_MOTION SOT_ADAPT_SF
export NVIDIA_TF32_OVERRIDE=0 PYTHONPATH=/home/airlab/user/kdh/SOT/pydeps CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1
cd /home/airlab/user/kdh/SOT; A=$PWD/analysis; D=/home/airlab/user/kdh/datasets/sot
python tools/collect_head_feats.py bio $D/biodrone_train $A/b1_bio_train_0.npz --shard 0/2 > logs/b1_bt0.log 2>&1 &
python tools/collect_head_feats.py bio $D/biodrone_train $A/b1_bio_train_1.npz --shard 1/2 > logs/b1_bt1.log 2>&1 &
python tools/collect_head_feats.py vis $D/visdrone_sot_trainval/train $A/b1_vis_train.npz > logs/b1_vt.log 2>&1 &
python tools/collect_head_feats.py bio $D/biodrone_val $A/b1_bio_val.npz > logs/b1_bv.log 2>&1 &
wait; echo ALLDONE

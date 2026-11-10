#!/bin/bash
cd /home/airlab/user/kdh/SOT/ORTrack/tracking
export PYTHONPATH=../../pydeps NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1 SOT_MOTION=cmc
export SOT_BIODRONE=/home/airlab/user/kdh/datasets/sot/biodrone_tune
for t in "0.7:10:replace:731" "0.8:20:replace:732" "0.7:10:mix:733"; do
  v=${t%:*}; id=${t##*:}
  SOT_TUPD=$v python test.py ortrack deit_tiny_patch16_224_uavft --dataset_name biodrone --runid $id --threads 8 --num_gpus 2 > /home/airlab/user/kdh/SOT/logs/tupd_$id.log 2>&1 &
done
wait
echo ALLDONE

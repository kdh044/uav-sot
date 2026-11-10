#!/bin/bash
# AVTrack(DeiT, D) / SGLATrack / LETrack 표준 4종 평가, TF32 off
export PYTHONPATH=/home/airlab/user/kdh/SOT/pydeps
export NVIDIA_TF32_OVERRIDE=0
run(){ cd /home/airlab/user/kdh/SOT/$1/tracking; for ds in $4; do echo "=== $1 $3 $ds $(date)"; python test.py $2 $3 --dataset_name $ds --threads 8 --num_gpus 2 2>&1 | grep -E "Done|Error" | tail -2; done; }
run AVTrack avtrack deit_tiny_patch16_224 "dtb70 uavdt visdrone2018 uav123"
run AVTrack avtrack deit_tiny_distilled_patch16_224 "dtb70 uavdt visdrone2018 uav123"
run SGLATrack sglatrack deit_distilled "dtb70 uavdt visdrone uav123"
run LETrack letrack baseline_WOCE "dtb70 uavdt visdrone uav123"
echo ALLDONE

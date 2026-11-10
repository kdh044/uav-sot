#!/bin/bash
# BioDrone val(100) 6개 모델 + ORTrack ad24, TF32 off
export PYTHONPATH=/home/airlab/user/kdh/SOT/pydeps NVIDIA_TF32_OVERRIDE=0
R=/home/airlab/user/kdh/SOT
go(){ cd $R/$1/tracking; echo "=== $1 $3 $(date +%T)"; python test.py $2 $3 --dataset_name biodrone $4 --threads 8 --num_gpus 2 2>&1 | grep -E "Done|Error" | tail -1; }
go ORTrack ortrack deit_tiny_patch16_224
go ORTrack ortrack deit_tiny_distilled_patch16_224
SOT_ADAPT_SF="24:3" go ORTrack ortrack deit_tiny_patch16_224 "--runid 24"
go AVTrack avtrack deit_tiny_patch16_224
go AVTrack avtrack deit_tiny_distilled_patch16_224
go SGLATrack sglatrack deit_distilled
go LETrack letrack baseline_WOCE
echo ALLDONE

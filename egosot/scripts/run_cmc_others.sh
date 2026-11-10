#!/bin/bash
# 다른 추적기에도 CMC 적용 (BioDrone val), 재학습 없음
export PYTHONPATH=/home/airlab/user/kdh/SOT/pydeps NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1 SOT_MOTION=cmc
R=/home/airlab/user/kdh/SOT
go(){ cd $R/$1/tracking; echo "=== $1 $3 $(date +%T)"; python test.py $2 $3 --dataset_name biodrone $4 --threads 8 --num_gpus 2 2>&1 | grep -E "Done|Error" | tail -1; }
go AVTrack avtrack deit_tiny_distilled_patch16_224 "--runid 77"
go ORTrack ortrack deit_tiny_distilled_patch16_224 "--runid 77"
go AVTrack avtrack deit_tiny_patch16_224 "--runid 77"
mv $R/SGLATrack/output/test/tracking_results/sglatrack/deit_distilled/biodrone $R/results_backup/sgla_bio_tmp_moved 2>/dev/null
go SGLATrack sglatrack deit_distilled ""
mv $R/LETrack/output/test/tracking_results/letrack/baseline_WOCE_298/biodrone $R/results_backup/letrack_bio_tmp_moved 2>/dev/null
go LETrack letrack baseline_WOCE ""
echo ALLDONE

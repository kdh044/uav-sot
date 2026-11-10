#!/bin/bash
# BioDrone 결과(<Tracker>_XXX.txt, 제출 형식)를 frame_XXX.txt 이름으로 링크한 채점용 폴더 생성
# usage: tools/bio_view.sh <result_dir(.../biodrone/result)> <tag>
src=$(realpath $1); dst=analysis/bio_views/$2; mkdir -p $dst
for f in $src/*_[0-9][0-9][0-9].txt; do id=$(basename $f .txt); id=${id##*_}; ln -sf $f $dst/frame_$id.txt; done
echo $dst

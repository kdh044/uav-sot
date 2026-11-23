# 시작하기

## Ours (학습 없는 버전)

1. 설치: `pip install -r tracker/requirements.txt`
2. 가중치: 공식 ORTrack-DeiT `ORTrack_ep0300.pth.tar` → `tracker/output/checkpoints/train/ortrack/deit_tiny_patch16_224/`
3. 데이터 경로: `tracker/lib/test/evaluation/local.py`
4. 실행: `cd tracker && bash run_looktwice.sh uav123`

## 통합 검증셋

```bash
python eval/make.py                    # 10종 × 30개 시퀀스
bash eval/run_ostrack.sh uav123 dtb70  # 비교용 OSTrack
python eval/score_ostrack.py           # AUC 표
```

## 코드 구성

| 경로 | 내용 |
|---|---|
| `tracker/lib/test/tracker/sm_track.py` | Ours: 두 위치 탐색 + 최소 박스 (`track_simple`) |
| `tracker/lib/test/tracker/motion_prior.py` | 카메라 움직임 추정 (특징점 흐름 + RANSAC 호모그래피) |
| `recovery_tracker/student.py` | 학생 추적기 |
| `recovery_tracker/query_decoder.py` | 오추적 판단 디코더 |
| `recovery_tracker/losses.py` | 정답 손실과 SAM3 증류 손실 |
| `recovery_tracker/train.py` | 학습 |

가중치와 데이터셋은 저장소에 포함하지 않습니다.

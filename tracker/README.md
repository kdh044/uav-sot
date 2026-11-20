# Look Twice tracker (ORTrack 기반)

공식 ORTrack(CVPR25) 코드에 학습 없는 추적 로직을 더한 것입니다. 가중치는 공식 것을 그대로 씁니다.

- **Look Twice**: 직전 위치와 카메라 움직임(CMC, LK 광류 + RANSAC 호모그래피)만큼 옮긴 위치를 배치 2로 함께 추적하고 점수가 높은 쪽을 고릅니다.
- **화면 밖 추적**: 표적이 화면 밖으로 나가면 위치만 호모그래피로 계속 옮기며 대기하고, 다시 들어오면 그 자리에서 찾습니다.

## 실행
1. 환경: `requirements.txt` (최신 torch에서 돌도록 일부 호환 수정 포함)
2. 가중치: 공식 ORTrack-DeiT `ORTrack_ep0300.pth.tar`
   ([Google Drive](https://drive.google.com/drive/folders/1pXq5HHezldjOyFCj6DUrXl5H8r4KiGv3)) →
   `output/checkpoints/train/ortrack/deit_tiny_patch16_224/`
3. 데이터 경로: `lib/test/evaluation/local.py`
4. 실행: `bash run_looktwice.sh uav123`
5. 채점: `python tracking/score.py <결과 폴더> uav123`

## 코드 위치
| 파일 | 내용 |
|---|---|
| `lib/test/tracker/sm_track.py` | `track_simple()` — Look Twice + 화면 밖 추적 |
| `lib/test/tracker/motion_prior.py` | CMC (특징점 추적 + 호모그래피) |
| `lib/test/tracker/ortrack.py` | env 로 위 로직을 켜는 진입점 (끄면 원본 ORTrack과 동일) |

`sm_track.py` 안의 다른 함수(`track_sm`, `track_beam`, `track_reacq`)와 다른 env 옵션은 실패한 실험 버전입니다. 최종본은 `track_simple`뿐입니다.

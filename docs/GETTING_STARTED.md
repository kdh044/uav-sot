# Getting started

## Look Twice

1. Install: `pip install -r tracker/requirements.txt`
2. Weights: official ORTrack-DeiT `ORTrack_ep0300.pth.tar` → `tracker/output/checkpoints/train/ortrack/deit_tiny_patch16_224/`
3. Data paths: `tracker/lib/test/evaluation/local.py`
4. Run: `cd tracker && bash run_looktwice.sh uav123`

## Unified validation set

```bash
python eval/make.py                    # 10 datasets × 30 sequences
bash eval/run_ostrack.sh uav123 dtb70  # OSTrack baseline
python eval/score_ostrack.py           # AUC table
```

## Code

| Path | Content |
|---|---|
| `tracker/lib/test/tracker/sm_track.py` | Look Twice (`track_simple`) |
| `tracker/lib/test/tracker/motion_prior.py` | camera motion (LK flow + RANSAC homography) |
| `recovery_tracker/student.py` | student tracker |
| `recovery_tracker/query_decoder.py` | identity decoder |
| `recovery_tracker/losses.py` | GT and SAM3 distillation losses |
| `recovery_tracker/train.py` | training |

Weights and datasets are not included.

<div align="center">

# KDI Tracker: Single Object Tracking for UAV Videos

DongHun Kim · Jeonbuk National University

[Results](docs/RESULTS.md) · [Getting started](docs/GETTING_STARTED.md)

</div>

<p align="center">
  <img src="assets/demo_biodrone.webp" width="100%">
</p>

<p align="center">
  <img src="assets/demo_vetra.webp" width="49%">
  <img src="assets/demo_viso.webp" width="49%">
</p>

<p align="center"><sub>Green: GT · Red: ORTrack · Blue: Look Twice (ours) — BioDrone (shake) · VETRA (1 fps) · VISO (tiny, zoomed)</sub></p>

## News

- **[2026/11/23]** Unified 10-dataset validation set: Look Twice 48.6 AUC vs. ORTrack 41.8 and OSTrack 43.1.
- **[2026/11/23]** SAM3-distilled student tracker implemented; full training in progress.

## Introduction

Given only the first-frame box, the tracker follows one target through drone,
manned-aircraft, satellite and thermal videos. We target three failures:

<p align="center">
  <img src="assets/failures.jpg" width="100%">
</p>

<p align="center"><sub>① drift to a similar object · ② target leaves the search area under camera shake · ③ tiny target</sub></p>

**Look Twice** (training-free) tracks both the previous position and the
camera-motion-compensated position, and keeps the more confident one.
The **student tracker** handles all three failures, distilled from SAM3 during training only.

<p align="center">
  <img src="assets/sam3-kd-structure.png" width="100%">
</p>

<p align="center">
  <img src="assets/auc_10datasets.png" width="80%">
</p>

## Getting started

1. Data: build the unified validation set with `python eval/make.py`.
2. Run: `cd tracker && bash run_looktwice.sh <dataset>`.
3. Evaluate: `python eval/score_ostrack.py`.

Details: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) · per-dataset results: [docs/RESULTS.md](docs/RESULTS.md).

## Acknowledgement

Built on [ORTrack](https://github.com/wuyou3474/ORTrack) and
[OSTrack](https://github.com/botaoye/OSTrack); teacher model
[SAM3](https://github.com/facebookresearch/sam3). Earlier Hailo / MOT work is on the
[`archive-hailo-mot`](https://github.com/kdh044/uav-sot/tree/archive-hailo-mot) branch.

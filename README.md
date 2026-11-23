<div align="center">

# KDI Tracker: Single Object Tracking for UAV Videos

DongHun Kim · Jeonbuk National University

</div>

<p align="center">
  <img src="assets/demo_biodrone.webp" width="100%">
</p>

<p align="center"><sub>Green: GT · Red: ORTrack · Blue: Look Twice (ours) · BioDrone, heavy camera shake</sub></p>

## Overview

Given only the target box in the first frame, the tracker follows that one target
to the end of the video, with no detector. It has to work across very different
footage: multirotor drones, flapping-wing drones, manned aircraft at 1 fps,
satellite video with targets of a few pixels, and thermal video.

## Problems

<p align="center">
  <img src="assets/failures.jpg" width="100%">
</p>

Lightweight trackers (e.g. ORTrack) fail in three ways on UAV video:

1. **Drift**: the box jumps to a similar object and keeps following it.
2. **Shake**: a sudden camera motion throws the target outside the search area.
3. **Tiny targets**: a few-pixel target disappears when the frame is downscaled.

## Method

**Look Twice** (training-free, done). For every frame the tracker searches two places
at once: where the target was, and where it should be if only the camera moved
(estimated from feature-point flow). It keeps whichever answer is more confident.
If the camera-motion estimate is wrong, the original position is still there,
so it rarely hurts. This addresses problem 2.

**Student tracker** (implemented, training). One network that addresses all three
problems, trained with SAM3 as a teacher. SAM3 is used only during training;
at test time only the small student runs.

| Problem | What the student does |
|---|---|
| Drift | For each candidate, decides "same target" or "no match". It updates its memory only when sure. SAM3's object memory teaches it what the target looks like. |
| Shake | Places several position guesses (camera-motion guess plus wider area) and picks one. A learned version of Look Twice. |
| Tiny targets | Reads fine details from the full-resolution frame and context from a downscaled frame, separately. |

<p align="center">
  <img src="assets/sam3-kd-structure.png" width="100%">
</p>

**Plan.** Starting from OSTrack, add the three modules one at a time
(shake → tiny targets → drift + SAM3), train each on GOT-10k, and check the gain
on the validation set below at every step.

## Results

### Quantitative

Unified validation set: 10 datasets × 30 sequences, Success AUC (%).

<p align="center">
  <img src="assets/datasets.jpg" width="100%">
</p>

| Dataset | Type | ORTrack | **Look Twice** | OSTrack (ViT-B) |
|---|---|---:|---:|---:|
| VETRA | manned aircraft, 1 fps | 8.21 | **42.75** | 9.42 |
| BIRDSAI | fixed-wing, thermal | 31.96 | **39.45** | 29.13 |
| BioDrone | flapping-wing drone | 34.37 | **40.92** | 40.57 |
| UAV123 | multirotor | 62.44 | 62.88 | **67.21** |
| DTB70 | multirotor | 70.85 | 70.99 | **72.05** |
| UAVDT | multirotor | 57.11 | 57.93 | **59.37** |
| VisDrone2018 | multirotor | **68.52** | 68.60 | 67.65 |
| SV248S | satellite, tiny | 18.30 | **25.16** | 16.21 |
| SatSOT | satellite, tiny | 34.77 | **37.15** | 37.12 |
| VISO | satellite, tiny | 31.49 | **39.75** | 32.19 |
| **Mean** | | 41.80 | **48.56** | 43.09 |

<p align="center">
  <img src="assets/auc_10datasets.png" width="80%">
</p>

Look Twice improves the mean by +6.8 over ORTrack and +5.5 over the larger OSTrack.
The gain comes from shaky, low-frame-rate and tiny-target videos; on ordinary
multirotor footage OSTrack is still 2 to 4 points higher.

<sub>SV248S: 28 sequences (2 have no first-frame GT, excluded for every method). Look Twice without off-screen tracking.</sub>

### Qualitative

| VETRA (1 fps, large jumps) | VISO (tiny car, zoomed) |
|---|---|
| ![VETRA](assets/demo_vetra.webp) | ![VISO](assets/demo_viso.webp) |
| AUC 0.3 → 79.7 | AUC 14.8 → 50.1 |

<sub>Green: GT · Red: ORTrack · Blue: Look Twice. Clips are the sequences with the largest gain. Full videos: [BioDrone](assets/demo_biodrone.mp4) · [VETRA](assets/demo_vetra.mp4) · [VISO](assets/demo_viso.mp4)</sub>

## Getting started

```bash
cd tracker && bash run_looktwice.sh uav123   # Look Twice (official ORTrack weights)
python eval/make.py                          # build the 10-dataset validation set
python eval/score_ostrack.py                 # AUC table
```

Setup and code layout: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## Acknowledgement

Built on [ORTrack](https://github.com/wuyou3474/ORTrack) and
[OSTrack](https://github.com/botaoye/OSTrack); teacher model
[SAM3](https://github.com/facebookresearch/sam3). Earlier Hailo / MOT work is on the
[`archive-hailo-mot`](https://github.com/kdh044/uav-sot/tree/archive-hailo-mot) branch.

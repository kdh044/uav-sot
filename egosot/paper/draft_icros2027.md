# 카메라 움직임 보정과 UAV·소형 객체 미세조정을 결합한 경량 ViT 기반 UAV 단일 객체 추적

김동훈 (전북대학교 AI 로보틱스 연구실)

## 요약
경량 ViT 기반 UAV 단일 객체 추적기(SOT)는 표준 벤치마크에서 AUC 63~65로 포화되었지만, 기체 흔들림과 회전이 큰 실제 비행 영상(BioDrone)에서는 AUC 25~28로 크게 떨어진다. 실패 사례를 분석한 결과, 표준 벤치마크에서는 실패의 약 10%에 불과하던 "검색 영역 밖 이탈"이 실제 비행 영상에서는 30~47%를 차지하였다. 본 논문은 (1) 배경 특징점 기반 호모그래피로 카메라 움직임을 추정해 검색 영역을 보정하는 CMC와 (2) UAV 영상과 소형 객체 영상을 함께 쓰는 미세조정을 결합한다. ORTrack(CVPR 2025)에 적용한 결과 BioDrone val AUC가 26.6에서 35.7로 향상되었고 추적 실패는 313회에서 166회로 47% 감소하였다. 소형 객체 데이터를 함께 쓰면 UAV 데이터만으로 미세조정할 때 생기는 극소형 대상(SatSOT) 성능 저하(38.0→30.0)가 사라졌으며(38.0), 2.2배 무거운 ViT-B 추적기(30.9)보다도 높은 성능을 얻었다.

## 1. 서론
- 드론 기반 SOT는 경량성이 핵심이며, 최근 연구(AVTrack, ORTrack, SGLATrack, LETrack)는 DeiT-Tiny 백본과 one-stream 구조를 공유한다 [1–5].
- 이 계열은 직전 위치 주변의 검색 영역(대상 크기 4배)만 보므로, 카메라가 크게 움직이면 대상이 영역 밖으로 벗어난다.
- 기존 연구는 일반 영상(LaSOT, GOT-10k 등)으로 학습하고 저속 회전익 벤치마크로 평가하여 이 문제가 드러나지 않았다.
- 기여: (i) 표준 벤치마크와 실제 비행 영상의 실패 원인 비교 분석, (ii) CMC와 UAV·소형 객체 미세조정의 결합 및 시너지 확인, (iii) 6개 경량 추적기에 대한 CMC 일반성과 동일 미세조정 공정 비교.

## 2. 방법
**기반 추적기.** ORTrack-DeiT: 템플릿 128×128(대상 2배), 검색 영역 256×256(대상 4배), 16px 패치 ViT.

**카메라 움직임 보정(CMC).** 직전 프레임과 현재 프레임을 폭 640px로 축소한 뒤, 대상 주변을 제외한 영역에서 Shi-Tomasi 특징점 [13]을 검출하고 피라미드 LK 광학 흐름 [14]으로 추적하여 RANSAC [15]으로 호모그래피 H를 추정한다. 직전 예측 중심을 H로 변환한 위치를 현재 검색 영역의 중심으로 사용한다. 추정이 퇴화하면 중심을 화면 안으로 제한한다.

**UAV·소형 객체 미세조정.** 공식 가중치에서 시작하여 BioDrone train 270개(튜닝용 30개 제외), VisDrone-SOT train 86개, LaTOT 433개 시퀀스를 1:1:1로 섞어 20 epoch 미세조정한다(LR 4×10⁻⁵, 백본 0.1배, 단일 GPU 약 23분). LaTOT는 극소형 대상(중앙값 13px)에 대한 표현을 유지하기 위해 넣었다. 평가 시퀀스는 학습에 포함하지 않는다.

## 3. 실험
**설정.** 표준 4종(UAV123, DTB70, UAVDT, VisDrone2018), BioDrone val(100개), SatSOT(위성 극소형, 105개)에서 OPE 방식으로 평가한다. 지표는 Success AUC, 추적 실패(IoU=0이 5프레임 이상 지속) 횟수, 실패 시 대상이 검색 영역 밖에 있던 비율이다. BioDrone은 대상이 보이는 프레임만 채점한다. 속도는 RTX PRO 6000 GPU에서 crop·전처리를 포함해 측정하였다.

**표 1. 경량 추적기 비교 (AUC)**

| 모델 | 표준 4종 | BioDrone | +CMC (BioDrone) | FPS |
|---|---:|---:|---:|---:|
| ORTrack-DeiT [1] | 64.8 | 26.6 | 31.3 | 308 |
| ORTrack-D [1] | 63.7 | 24.8 | 33.3 | 446 |
| AVTrack-DeiT [2] | 63.7 | 24.5 | 32.4 | 322 |
| AVTrack-D [2] | 63.5 | 25.4 | 33.9 | 485 |
| SGLATrack [3] | 63.3 | 24.9 | 32.6 | 386 |
| LETrack [4] | 63.7 | 27.5 | 34.2 | 174 |
| OSTrack ViT-B [5] | (UAV123 69.0) | 30.9 | – | 139 |
| AVTrack-DeiT + UAV FT | 64.5† | 24.2 | 32.7 | 322 |
| AVTrack-D + UAV FT | 59.7† | 24.3 | 31.2 | 485 |
| **제안 (ORTrack + UAV·소형 FT + CMC)** | 63.7 | 27.8 | **35.7** | 124 |

† UAV123만 평가. FPS는 CMC 제외 값(제안만 CMC 포함).

**표 2. Ablation (ORTrack-DeiT)**

| CMC | 미세조정 데이터 | BioDrone AUC | 실패 (안/밖) | SatSOT AUC | 표준 4종 |
|:-:|---|---:|---:|---:|---:|
| | – | 26.6 | 313 (170/143) | 38.0 | 64.8 |
| ✓ | – | 31.3 | 200 (107/93) | 37.4 | 64.7 |
| | UAV | 28.2 | 275 (129/146) | 30.9 | 63.4 |
| ✓ | UAV | 35.3 | 163 (61/102) | 30.0 | 63.2 |
| | UAV + 소형 | 27.8 | 268 (116/152) | – | 65.7‡ |
| ✓ | UAV + 소형 | **35.7** | 166 (77/89) | **38.0** | 63.7 |

‡ UAV123만 평가.

**분석.**
- CMC는 6개 추적기 모두에서 BioDrone +4.7~+8.6의 향상을 보였고, 표준 4종에서는 손해가 없었다(64.8→64.7). 카메라가 거의 고정된 위성 영상(SatSOT)에서도 영향이 작았다(−0.6).
- 미세조정 단독은 +1.2~+1.6이지만 CMC와 결합하면 +8.7~+9.1로 개별 효과의 합을 넘는다. CMC가 대상을 검색 영역 안에 유지하고, 미세조정이 영역 안의 유사 물체 혼동을 줄이기 때문이다(영역 안 실패 170→61~77).
- UAV 데이터만으로 미세조정하면 극소형 대상(SatSOT)에서 AUC가 38.0→30.0으로 떨어지지만, 소형 객체 데이터(LaTOT)를 함께 쓰면 38.0으로 회복되고 16px 미만 구간은 원본보다 높아진다(8px 미만 mIoU 0.137→0.174).
- 프레임 간격을 3배로 늘린 UAV123@10fps에서는 CMC 효과가 없었다(65.3→65.4). 영역 밖 실패가 9~11건에 불과해, CMC의 이득은 프레임률이 아니라 기체 자체의 움직임 크기에 좌우된다.
- 해상 도메인(SeaDronesSee val 70클립)에서도 원본 72.7 → 최종 74.2로 손해 없이 소폭 향상되었다.
- 공정 비교: AVTrack 두 모델에 같은 UAV 미세조정을 적용해도 AUC는 31.2~32.7로 제안 방법(35.7)이 가장 높다.
- 한계: 미세조정 데이터 구성(UAV vs UAV+소형)은 튜닝 분할(30개)에서는 UAV만 쓴 쪽이 높았으나(43.3 vs 39.2), 소형 객체 일반화라는 설계 목적에 따라 UAV+소형을 택하였다. 표준 4종 평균은 1.1 감소하였다(64.8→63.7). CMC는 5.1 ms를 추가한다. 남은 실패의 54%는 여전히 영역 밖 이탈이며, 카메라 회전이 클 때 집중된다.

(그림 1: 실패 원인 `figs/paper/fig_failure_cause.png`, 그림 2: 속도-정확도 `figs/paper/fig_speed_acc.png`, 정성 비교 `figs/final_compare/`)

## 4. 결론
실제 비행 영상에서 경량 ViT 추적기의 주요 실패 원인이 카메라 움직임에 따른 검색 영역 이탈임을 보이고, CMC와 UAV·소형 객체 미세조정의 결합으로 AUC +9.1, 실패 47% 감소를 얻으면서 극소형 대상 성능도 유지하였다. 향후 회전에 강한 표현 학습과 Hailo-8 NPU 실기 배포를 연구한다.

## 참고문헌
[1] Y. Wu, X. Wang, X. Yang, M. Liu, D. Zeng, H. Ye, and S. Li, "Learning occlusion-robust vision transformers for real-time UAV tracking," in Proc. CVPR, 2025.
[2] Y. Wu et al., "Learning an adaptive and view-invariant vision transformer for real-time UAV tracking," in Proc. ICML, 2024.
[3] "Similarity-guided layer-adaptive vision transformer for UAV tracking," in Proc. CVPR, 2025. (저자 확인 필요)
[4] "Toward low-cost yet effective temporal learning for UAV tracking," in Proc. CVPR, 2026. (저자 확인 필요)
[5] B. Ye, H. Chang, B. Ma, S. Shan, and X. Chen, "Joint feature learning and relation modeling for tracking: A one-stream framework," in Proc. ECCV, 2022.
[6] H. Touvron et al., "Training data-efficient image transformers & distillation through attention," in Proc. ICML, 2021.
[7] M. Mueller, N. Smith, and B. Ghanem, "A benchmark and simulator for UAV tracking," in Proc. ECCV, 2016.
[8] S. Li and D.-Y. Yeung, "Visual object tracking for unmanned aerial vehicles: A benchmark and new motion models," in Proc. AAAI, 2017.
[9] D. Du et al., "The unmanned aerial vehicle benchmark: Object detection and tracking," in Proc. ECCV, 2018.
[10] P. Zhu et al., "Detection and tracking meet drones challenge," IEEE TPAMI, 2021.
[11] X. Zhao et al., "BioDrone: A bionic drone-based single object tracking benchmark for robust vision," IJCV, 2024.
[12] Y. Zhu et al., "Tiny object tracking: A large-scale dataset and a baseline," IEEE TNNLS, 2024. (LaTOT)
[13] J. Shi and C. Tomasi, "Good features to track," in Proc. CVPR, 1994.
[14] B. D. Lucas and T. Kanade, "An iterative image registration technique with an application to stereo vision," in Proc. IJCAI, 1981.
[15] M. A. Fischler and R. C. Bolles, "Random sample consensus," Commun. ACM, 1981.
[16] N. Aharon, R. Orfaig, and B.-Z. Bobrovsky, "BoT-SORT: Robust associations multi-pedestrian tracking," arXiv:2206.14651, 2022.
[17] "SatSOT: A benchmark dataset for satellite video single object tracking," IEEE TGRS, 2022. (저자 확인 필요)

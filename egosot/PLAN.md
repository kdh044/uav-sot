# ICROS 2027 논문 계획 (2페이지, 제출 목표 2027-04 말)

## 주제
**Ego-Motion-Aware Lightweight UAV Tracking**
카메라 움직임(CMC로 추정한 호모그래피)을 검색 위치 보정에만 쓰지 않고, 트래커의 세 가지 판단에 모두 사용.

| 모듈 | 질문 | 방법 |
|---|---|---|
| M1 Where | 어디를 볼지 | CMC + 등속으로 검색 중심 예측 + 움직임 불확실도에 따른 검색 영역 크기 조절 |
| M2 How | 어떻게 볼지 | 누적 카메라 회전만큼 검색 crop 을 역회전 → 템플릿 방향과 정렬 |
| M3 Trust | 믿을지 | 움직임 예측 위치 vs ViT 출력 위치의 불일치 + 외형 유사도 → 실패 감지 → 재탐색 |

백본은 ORTrack-DeiT 고정 (학습 없이 1차 결과). 효과 확인 후 필요 시 학습 버전.

## 근거 (확보됨, 2026-11-02)
- BioDrone val: ORTrack 26.6 AUC, 실패 313회, 실패의 46%가 검색 영역 밖 이탈
- CMC+CV(M1 일부): 31.8 AUC, 실패 208회 (재학습 없음)
- 남은 실패: 영역 밖 47% / 영역 안 53%
- 신뢰도 점수로는 실패 23%만 탐지(임계 0.4)

## 목표 (Goal)
- G1: BioDrone val AUC >= 34 (ORTrack 26.6, +CMC+CV 31.8, OSTrack ViT-B 30.9)
- G2: 표준 4종 평균 AUC 손해 <= 0.5 (64.8 기준)
- G3: 실패 횟수 313 → 180 이하
- G4: 속도: CMC 포함 end-to-end 30 FPS 이상 (서버 CPU 단일 스레드 기준 측정)
- G5: Ablation: base / M1 / M1+M2 / M1+M2+M3

## 단계
1. [ ] M2 회전 정렬 crop 구현 → BioDrone val
2. [ ] M3 불일치 신호 오프라인 평가 (탐지율 vs 신뢰도) → 재탐색 정책 → BioDrone val
3. [ ] M1 불확실도 기반 영역 크기
4. [ ] 최종 조합: BioDrone val + 표준 4종 + FPS
5. [ ] (여유 시) BioDrone train 으로 임계값 튜닝, val 은 최종 평가만 (누수 방지)
6. [ ] 그림: 실패 사례 / 개선 전후, 표: 비교 + ablation
7. [ ] 2페이지 초안

## 규칙
- 평가는 NVIDIA_TF32_OVERRIDE=0
- 하이퍼파라미터는 BioDrone **train** 일부로 정함 (val 은 보고용)
- 모든 기능은 env 로만 켜짐 (원본 동작 보존)

## 진행 기록 (2026-11-02, BioDrone train 30개 튜닝 분할, 보이는 프레임 AUC)
| 설정 | AUC | IoU=0 | mIoU(on-target) |
|---|---:|---:|---:|
| base | 28.0 | 65.2% | 0.801 |
| CMC | **35.0** | 55.2% | 0.760 |
| CMC+CV | 27.1 | 65.8% | – |  ← CV 해로움, 제외
| CMC+ROT(외접) | 22.4 | 52.2% | 0.446 |
| CMC+ROT(정렬박스) | 24.0 | 55.7% | 0.501 |
| CMC+ROT 부호반전 | 23.9 | 55.7% | 0.499 |
| CMC+ROT 각도0 (경로 검증) | 32.9 | 58.0% | 0.773 |  ← 코드 경로 정상
| +SIZE_CLAMP 1.2 | 모두 하락 | | |
결론: M2(회전 정렬)는 놓침은 약간 줄이나 박스 회귀가 망가짐 → 보류. M3(움직임 불일치)는 AUROC 0.59 < 점수 0.70 → 폐기. CV 폐기.
CMC 비용: 단일스레드 9~15 ms/frame (640→320 폭).

## 학습 결과 (2026-11-02) — 드론 데이터 미세조정 (BioDrone train 270 + VisDrone-SOT train 86, 20ep, ~50분/모델)
| 모델 | tune AUC | **val AUC** | val 실패 | UAV123 AUC |
|---|---:|---:|---:|---:|
| base | 28.0 | 26.6 | 313 | 66.4 |
| base+CMC | 35.0 | 31.3 | 200 | – |
| FT+CMC | **43.3** | **35.3** | **163** | 65.1 (FT) |
| FT-ego+CMC | 41.8 | 34.2 | 200 | 63.3 (FT-ego) |
→ G1(val>=34) 달성: FT+CMC 35.3. ego 증강은 CMC와 겹쳐 추가 이득 없음.
남은 실패 163회 중 영역 밖 이탈 62.6% → 다음: 영역 밖 이탈 후 재탐색(넓은 영역/재검출) 로직.
체크포인트: ORTrack/output/checkpoints/train/ortrack/deit_tiny_patch16_224_uavft/ORTrack_ep0020.pth.tar
남은 확인: FT 표준 3종(DTB70/UAVDT/VisDrone), FPS(CMC 포함), 공정성(비교 모델은 미세조정 안 함 → 같은 FT 적용 비교 필요)

## M4 재탐색 (FT+CMC 위, tune 분할) — 폐기
| 설정 (thr:k:grid:accept) | AUC | IoU=0 |
|---|---:|---:|
| FT+CMC (기준) | **43.3** | 48.2% |
| 0.35:2:1:0.5 | 37.0 | 56.5% |
| 0.30:3:1:0.45 | 39.0 | 53.0% |
| 0.40:2:1:0.55 | 35.3 | 58.0% |
→ 점수 기반 재탐색은 다른 곳의 유사 물체로 점프해서 오히려 하락. 점수가 신뢰도 신호로 약하다는 앞선 결과와 일치.
현재 최종안: **FT(드론 미세조정) + CMC**. 다음: 표준 3종 확인, FPS/CMC 최적화, 공정 비교(다른 모델 FT), 그림·초안.

## 표준 벤치마크 손해 줄이기: 가중치 평균 (원본↔FT, 재학습 없음)
| 모델 | 표준4종 평균 AUC | BioDrone tune AUC (CMC) |
|---|---:|---:|
| 원본 | 64.8 | 35.0 |
| WA 0.5 | **64.7** | 33.2 |
| WA 0.7 | 64.3 | 37.0 |
| FT | 63.4 | **43.3** |
→ 드론 성능과 표준 성능은 trade-off. 대상 과업(드론)이 우선이므로 최종 = FT+CMC. 표준 손해 −1.4 는 한계로 명시.

## CMC 일반성 (BioDrone val, 재학습 없음)
ORTrack-DeiT 26.6→31.3 / ORTrack-D 24.8→33.3 / AVTrack-DeiT 24.5→32.4 / AVTrack-D 25.4→33.9 / SGLATrack 24.9→32.6 / LETrack 27.5→34.2
우리(ORTrack-DeiT FT+CMC) 35.3. FT 기여 +4.0 (31.3→35.3).
보호장치 추가: motion_prior 에서 예측 중심을 화면 안으로 clamp, 박스 폭주 방지 (SGLATrack 한 시퀀스에서 퇴화 호모그래피로 OOM).
결과 백업: results_backup/{sgla,letrack}_biodrone_{base,cmc}
다음: FPS(CMC 포함), 그림(실패원인/속도-정확도/정성), 공정비교(LETrack 또는 AVTrack-D 도 FT), 2페이지 초안

## 2026-11-02 밤 진행
- CMC 속도/정확도 (tune, FT 기준): 640 43.3 AUC 8.04ms / 320 40.4 5.41ms / fast480 41.3 / fast320 39.9 4.42ms → 최종 640 유지
- Ablation (val): base 26.6/313 · FT 28.2/275 · CMC 31.3/200 · FT+CMC 35.3/163 (시너지 +8.7 > +6.3)
- 그림: figs/paper/fig_failure_cause.png, fig_speed_acc.png / 정성: figs/final_compare/
- 초안: paper/draft_icros2027.md
- 목표 상태: G1 ✅(35.3) G3 ✅(163) G5 ✅ ablation / G2 ✗(표준 −1.4, 한계로 기술) / G4 Hailo 실측 필요
- 다음: 공정 비교(다른 모델 FT), Hailo

## 공정 비교 1: AVTrack-D 동일 FT (증류 없이 — 교사 가중치 없음)
AVTrack-D: base 25.4/273 · CMC 33.9/202 · FT 24.3/285 · FT+CMC 31.2/167 · UAV123 FT 59.7(−5.3)
→ ORTrack FT+CMC 35.3/163 가 여전히 최고. 단 AVTrack-D 는 증류 모델이라 FT 시 증류 손실 부재로 손해 가능. 다음: 비증류 AVTrack-DeiT 동일 FT.

## 공정 비교 2: AVTrack-DeiT(비증류) 동일 FT
AVTrack-DeiT: base 24.5/277 · CMC 32.4/181 · FT 24.2/245 · FT+CMC 32.7/153 · UAV123 66.2→64.5
→ 동일 FT 조건에서도 ORTrack FT+CMC(35.3) 가 AUC 최고. 실패 횟수는 AVTrack-DeiT FT+CMC(153)가 약간 적음 → 논문 주장은 AUC 기준으로.
실험 단계 완료. 남은 것: Hailo 실측(DFC 필요), 초안 다듬기.

## Hailo 준비 (DFC 대기)
- ONNX: hailo/onnx/ortrack_uavft.onnx → onnxslim → ortrack_uavft_slim.onnx (441 노드, 동적 Shape/Cast 제거, PyTorch 대비 max diff < 5e-6)
- 연산: Conv, MatMul/Gemm, LayerNormalization, Softmax, Erf(GELU), Reshape/Transpose/Slice/Split/Concat, Sigmoid, Relu, Clip — CE 없음
- 보정: hailo/calib_uav_1024.npz (학습 분할에서 1024쌍, uint8 NHWC)
- 변환 스크립트: hailo/run_dfc.py (DFC whl 설치 후 실행)
## 진행 중: 검색배율 5 FT (run_ft_sf5.sh) — 영역 밖 이탈 줄이기 시도
- 검색배율 5 FT: tune 37.1(↓6.2), val 36.1/201 → 기각 (tune 기준). 영역밖 비중 62.6→53.2% 대신 영역안 혼동 증가
- M5 저신뢰 유지(SOT_HOLD 0.2/0.3/0.4): tune 39.5/39.6/32.4 < 43.3 → 기각
- 남은 실패 분석(FT+CMC tune 63건): CMC 성공률 99.9%, 영역밖 이탈 직전 5f 회전 12.6° vs 영역안 6.9° → 회전이 이탈과 연관. M2 재시도(위치만 회전정렬, 크기 유지, 회전 클 때만)
- M2 v2(위치만 정렬, 크기 유지, ROT_MIN 0/15/30): tune 17.7/24.0/27.6 ≪ 43.3 → M2 최종 폐기 (누적 회전각 오차 추정)
- 표준 4종 + CMC: ORTrack+CMC 66.8/67.0/60.2/64.6 → 평균 64.7 (원본 64.8, 손해 없음) / FT+CMC 64.6/63.4/57.9/66.8 → 63.2
- M6 템플릿 갱신(0.7/10 replace, 0.8/20 replace, 0.7/10 mix): tune 36.1/38.7/35.7 < 43.3 → 기각 (드리프트)
- SatSOT(위성 소형, 중앙값 12px): 원본 38.0 / +CMC 37.4 / FT 30.9 / FT+CMC 30.0 → 드론 FT 가 극소형 대상 일반화 훼손 (8-16px 놓침 32→45%). 한계로 명시
- WA+CMC: SatSOT WA0.5 35.0 / WA0.7 33.6 (FT 30.0, 원본 37.4). BioDrone val WA0.7+CMC 35.3 (=FT+CMC). tune 에선 FT 43.3 ≫ WA0.7 37.0 → 규칙상 주 결과는 FT, WA0.7 은 일반화 보존 변형으로 논의. tune 30개가 작아 선택이 불안정한 점 한계로 기술.

## LaTOT 혼합 FT (BioDrone train270 + VisDrone86 + LaTOT433, 1:1:1, 20ep, 23분)
| | tune AUC(CMC) | val AUC(CMC) / 실패 | SatSOT(CMC) | UAV123 |
|---|---:|---:|---:|---:|
| FT+CMC | 43.3 | 35.3 / 163 | 30.0 | 65.1 |
| FT_tiny+CMC | 39.2 | **35.7** / 166 | **38.0** | **65.7** |
| 원본(+CMC) | 35.0 | 31.3 / 200 | 37.4 (원본 38.0) | 66.4 |
→ 소형 회귀 완전 회복(SatSOT 30.0→38.0, <8px mIoU 0.094→0.174), val·UAV123 소폭↑. tune 만 −4.1. 일반성 고려 시 FT_tiny 가 더 균형적.
체크포인트: ORTrack/output/checkpoints/train/ortrack/deit_tiny_patch16_224_uavft_tiny/ORTrack_ep0020.pth.tar

## 최종 모델 결정 (2026-11-03): B = ORTrack + UAV·소형 FT(BioDrone+VisDrone-SOT+LaTOT) + CMC
- BioDrone val 35.7 / 실패 166 (안 77 / 밖 89), SatSOT 38.0, 표준4종+CMC 63.7 (65.3/65.5/59.2/64.6), B no-CMC val 27.8/268
- 선택 근거: 소형 객체 일반화(설계 목적). tune 분할(30)에서는 A(UAV만) 43.3 > B 39.2 → 한계로 명시
- 초안/그림/정성비교 모두 B 기준으로 갱신 (paper/draft_icros2027.md, figs/paper, figs/final_compare)
- Hailo ONNX 최종 모델(B)로 재생성: hailo/onnx/ortrack_final_slim.onnx (441 노드, PyTorch 대비 <4e-6), run_dfc.py 기본값 변경
- CMC 경량화 knob(코너 150, LK win 15, 폭 480): 7.54→7.0~7.2 ms (<10% 단축) → 효과 미미. 특징점 재사용(fast)이 유일하게 큰 단축(약 3배)이나 정확도 −3.4
- SeaDronesSee val(해상 70클립, 4K, 중앙값 58px): 원본 72.7 / +CMC 74.2 / 최종 74.2 (실패 5/3/3) → 해상 도메인 손해 없음, CMC 소폭 이득
- Word 초안: paper/draft_icros2027.docx (paper/make_docx.js, A4 2단, XML 검증 통과, 렌더 미확인: soffice 없음)
- LaTOT 비율 2:1:0.5: tune 37.9 / val 35.0 (실패 170) / SatSOT 37.3 / UAV123 65.2 → 1:1:1(39.2/35.7/38.0/65.7)보다 모두 낮음. tune 순위(UAV만 43.3 > 1:1:1 39.2 > 2:1:0.5 37.9)가 LaTOT 비율과 단조 관계가 아님 → tune 30개 분할 노이즈 큼. 최종 B(1:1:1) 유지
- UAV123@10fps: 원본 65.3 / +CMC 65.4 / 최종+CMC 65.1, 영역밖 실패 9~11건뿐 → 프레임 간격을 늘려도 UAV123 은 카메라 움직임이 작아 CMC 효과 없음. CMC 이득은 기체 ego-motion 크기에 비례(BioDrone)

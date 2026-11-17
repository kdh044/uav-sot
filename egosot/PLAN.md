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

## 2026-11-11: 다음 연구 방향을 위한 통제 진단 (학습 없음)

- 원문 검토: `paper/LITERATURE_REVIEW_20261111.md`. 단순 CMC, 가변 검색, 고저층 특징 결합, temporal memory는 직접 겹치는 선행 연구가 있어 신규성으로 단정하지 않음.
- BioDrone tune 30개, 고정 900개 transition × 10조건 × 원본/기존 FT. 이전 GT crop 및 template oracle이 포함된 진단으로 공식 OPE AUC가 아님. TF32 off, FP32.
- 같은 clip·같은 4,597 유효 프레임 비교(mIoU, 시퀀스 평균): 원본 free 55.5 / 이전 GT 위치·크기 crop 79.0 / +최근 GT template 82.7. FT 57.5 / 81.3 / 83.4. FT crop oracle +23.9%p, 최근 template 추가 +2.0%p. 실제 배포 개선량이 아님.
- FT free에서 GT–crop 겹침 없음은 시퀀스 평균 30.0%, GT crop에서는 3.0%. drift 이후 프레임도 포함해 실패 시작 원인 비율과 구분.
- 회전 대조(공통 760 transition): FT clean 83.6 → search +30° 52.4 mIoU. 중심 적중률@원래 대상크기의 .25는 96.6→95.6%, GT AABB 면적 1.92배/예측면적비 .55. template 동시 회전 이득 +.3%p(CI -1.0~+1.4) → 회전 매칭보다 label/box extent 영향이 커 단순 등변 matching의 근거로 쓰지 않음.
- source <16px 포함 표본 25개(12 seq), 모델 입력 대상 중앙값 61.9px → 원본 크기와 input token 수를 구분. 이 표본만으로 소형 표현 방법을 확정하지 않음.
- 산출물: `analysis/matching_20261111/FINDINGS.md`, `REPORT.md`, `ROTATION_REPORT.md`, `ROLLOUT_REPORT.md`, PNG/PDF, 프레임별 JSONL, metadata와 source snapshot.
- 다음 가설: 실제 예측 오차가 누적되는 상태의 visual tracking 학습. 완전히 crop 밖인 대상에는 관측 범위/재획득 정보도 필요하므로 local loss만으로 해결된다고 주장하지 않음. 새 방법 학습은 아직 하지 않음.

## 2026-11-11: 극소형 대상 독립 진단 (SatSOT / VISO, 학습 없음)

- SatSOT 105 seq + VISO 44 scene에서 scene당 1 tracklet(고정 seed, median size<16 우선). 중간 GT 초기화 최대120f clip. 모델/조건마다 동일 유효17,494f; 공식 OPE AUC가 아님.
- 핵심 교란요인: 원본 ORTrack/LETrack clip_box(...,margin=10)가 w,h 최소10px를 강제. 원본 파일 git 변경 없음 확인. 경계 clipping을 유지하고 w,h 최소값만1로 바꾼 별도 프로세스 비교.
- 원본 크기<8px에서 mIoU min10→floor1: ORTrack Sat23.4→40.1 / VISO20.5→39.0; 기존 FT Sat29.0→48.9 / VISO23.2→45.6; LETrack Sat27.2→41.2 / VISO24.1→41.9. 후처리 통제이며 새 알고리즘 기여가 아님.
- floor1 동일 sampled frame FT: Sat 실제연속53.8 / GT 이전crop64.2 / +최근GT template72.6; VISO54.9 /62.5 /69.7. GT template는 배포 불가능 oracle. 기존 BioDrone만으로 외형/시간 정보의 중요성을 낮게 판단하면 안 됨.
- LETrack floor1 history유지−매프레임초기화: Sat+.45%p(CI -1.12~+1.95), VISO-.71(CI -2.00~+.53). 추론 ablation이며 no-history 재학습 비교는 아님.
- 과거1/5/10 GT template 예측 box 단순평균은 최근GT 하나보다 뚜렷한 우위 없음. temporal 학습 전체를 부정하는 결과는 아님.
- 다음 후보: 최근GT template의 이득과 실제예측 template의 drift 사이 간극을 줄이는 학습. HIPTrack/PATT/KeepTrack/LETrack 대비 신규성 검토 필요.
- `analysis/tiny_20261111/FINDINGS.md`, `REPORT.md`, `summary.json`, raw JSONL, manifest, source snapshot, 최소박스 비교 사례 PNG/PDF. 기존 SatSOT AUC38.0을 이 진단 수치로 교체하지 않음. 새 학습·기존 코드/가중치 변경·외부 업로드 없음.


### 2026-11-11 — 확장 문헌·원문 보관

사용자 요청으로 문헌 목록·분석·원문 자료를 저장. paper/RESEARCH_INDEX_20261111.md 참조. 66편 목록/CSV, 검색 영역 5편 보충, PDF 21개 및 추출 텍스트·출처·해시 보관. 관련 방법 절 확인은 18편이며 전 논문 정독/재현을 뜻하지 않음. GitHub 업로드는 수행하지 않음.

## 2026-11-12: rollout-matched crop 학습 (exposure bias 재현, 기여 아님)
- `analysis/rollout_train_20261112/README.md`. uavft_tiny+CMC 의 train270 rollout crop 분포(중심 crop 안 50.7%, crop-miss 47.3%)를 bank 로, search crop 50% 를 bank 에서 배치(밖이면 cls 0·박스 loss 제외). Re3(RA-L18)·SLT(ECCV22) 효과 재현.
- val+CMC AUC: ctl 33.6/34.5 → rm50 38.8/37.6, UAV123 65–67 유지, crop-miss ~50→42–45%. tune+CMC 38.4/33.2 → 39.6/39.5. tune 무CMC 이득 없음(bank 가 CMC rollout 기반).
- 실패 건수 증가(163/172→186/190)는 재획득 증가 때문: 회복 건 81/88→114/113, 미회복 82/84→72/77, 실패 프레임 30.8k/30.2k→26.6k/27.6k. 실패=IoU0 5f+ 구간 "건수"라 쪼개짐이 많을수록 늘어남 → 이후 실패 프레임·미회복 건수 함께 보고.
- rm50 = 다음 실험(주변시/중심시)의 기준선.

## 2026-11-12: 실패 원인 분해 (전 데이터셋, uavft_tiny 무CMC, 학습 없음)
- `analysis/failure_decomp_20261112/README.md`. 실패 = IoU<.1 5f+ (lock IoU≥.5 이후), 시작 프레임의 실제 search crop 대비 OUT/PARTIAL/IN 분류 + 크기·ego(homography)·blur(Laplacian). 극소형은 floor1 주결과. TF32 off, GPU 1개 단일 프로세스.
- crop 이탈(OUT+PARTIAL) 비율: BioDrone val 73%(CI 64–82, +CMC 61%), tune 76%, BIRDSAI 51%, UAV123 25%, DTB70 17%, UAVDT 9%, VisDrone 0%, SatSOT 17%, SV248S 13%, VISO 17%, LaTOT(학습셋) 21%. SeaDronesSee 7건은 프레임 불연속 의심으로 제외.
- BioDrone OUT 시작에서 배경 이동 = crop 반폭의 1.14배(기준선 .20) → ego-motion 이 이탈 원인. CMC 후 남은 사건의 74%는 대상 비가시 구간 뒤 재등장.
- 소형: UAV 데이터에서 <16px 프레임 실패율 10–20배지만 사건의 13–19%뿐. 위성/극소형은 크기 구간별 실패율 평평, 실패는 crop 안 국소 오정합(예측 중심이 대상 1–2배 거리). blur는 어느 데이터에서도 주원인 아님(BioDrone 0.7–0.8배).
- 함의: 주변시(전체 프레임 저해상도)+중심시 구조는 BioDrone 실패의 61–76%(주로 재획득), 표준 UAV·위성 실패의 ≤25%만 다룸. 나머지는 중심시 분별/정합 문제.

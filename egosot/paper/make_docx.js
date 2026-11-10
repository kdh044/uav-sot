// ICROS 2027 2페이지 초안 (A4, 제목·요약 1단 + 본문 2단). node paper/make_docx.js
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, Table, TableRow, TableCell, WidthType,
  BorderStyle, ImageRun, SectionType, HeadingLevel, ShadingType,
} = require('docx');

const ROOT = '/home/airlab/user/kdh/SOT';
const FONT = '맑은 고딕';
const sz = (pt) => pt * 2;
const P = (text, o = {}) => new Paragraph({
  alignment: o.align || AlignmentType.JUSTIFIED, spacing: { after: o.after ?? 60, line: 260 },
  children: (Array.isArray(text) ? text : [text]).map((t) => typeof t === 'string'
    ? new TextRun({ text: t, font: FONT, size: sz(o.size || 9), bold: o.bold }) : t),
});
const H = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2, spacing: { before: 120, after: 60 },
  children: [new TextRun({ text, font: FONT, size: sz(10), bold: true })],
});
const B = (t) => new TextRun({ text: t, font: FONT, size: sz(9), bold: true });
const T = (t) => new TextRun({ text: t, font: FONT, size: sz(9) });

// 2단 폭 (A4 11906 DXA, 좌우 여백 1134, 단 간격 340) → 단 폭 ≈ 4649
const COLW = 4649;
function table(header, rows, widths, boldLast = true) {
  const border = { style: BorderStyle.SINGLE, size: 4, color: '808080' };
  const cell = (t, w, i, isHead, bold) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    margins: { top: 20, bottom: 20, left: 50, right: 50 },
    shading: isHead ? { type: ShadingType.CLEAR, fill: 'EDEDED', color: 'auto' } : undefined,
    borders: { top: border, bottom: border, left: border, right: border },
    children: [new Paragraph({
      alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [new TextRun({ text: t, font: FONT, size: sz(7), bold: isHead || bold })],
    })],
  });
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ children: header.map((h, i) => cell(h, widths[i], i, true)) }),
      ...rows.map((r, ri) => new TableRow({
        children: r.map((c, i) => cell(c, widths[i], i, false, boldLast && ri === rows.length - 1)),
      }))],
  });
}
const cap = (t) => P(t, { align: AlignmentType.CENTER, size: 8, after: 80 });
const img = (file, w) => {
  const png = fs.readFileSync(path.join(ROOT, file));
  const W = png.readUInt32BE(16), Hh = png.readUInt32BE(20);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new ImageRun({ type: 'png', data: png, transformation: { width: w, height: Math.round(w * Hh / W) } })],
  });
};

const head = [
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: '카메라 움직임 보정과 UAV·소형 객체 미세조정을 결합한 경량 ViT 기반 UAV 단일 객체 추적', font: FONT, size: sz(14), bold: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: 'Lightweight ViT-based UAV Single Object Tracking with Camera Motion Compensation and UAV/Tiny-Object Fine-tuning', font: FONT, size: sz(10), italics: true })],
  }),
  P('김동훈 (전북대학교 AI 로보틱스 연구실)', { align: AlignmentType.CENTER, size: 9, after: 160 }),
  P([B('요약 '), T('경량 ViT 기반 UAV 단일 객체 추적기는 표준 벤치마크에서 AUC 63~65로 포화되었지만, 기체 흔들림과 회전이 큰 실제 비행 영상(BioDrone)에서는 AUC 25~28로 크게 떨어진다. 실패 사례 분석 결과, 표준 벤치마크에서 실패의 약 10%였던 검색 영역 밖 이탈이 실제 비행 영상에서는 30~47%를 차지하였다. 본 논문은 호모그래피 기반 카메라 움직임 보정(CMC)과 UAV·소형 객체 영상을 함께 쓰는 미세조정을 결합한다. ORTrack(CVPR 2025)에 적용한 결과 BioDrone val AUC가 26.6에서 35.7로, 추적 실패는 313회에서 166회로 개선되었고, 극소형 대상(SatSOT) 성능은 유지되었다(38.0). 2.2배 무거운 ViT-B 추적기(30.9)보다도 높다.')], { after: 60 }),
  P([B('Keywords: '), T('UAV tracking, single object tracking, camera motion compensation, vision transformer, fine-tuning')], { after: 120 }),
];

const body = [
  H('1. 서론'),
  P('최근 경량 UAV 추적기(AVTrack [2], ORTrack [1], SGLATrack [3], LETrack [4])는 DeiT-Tiny [6] 백본의 one-stream 구조 [5]를 공유하며, 직전 위치 주변의 검색 영역(대상 크기 4배)만 본다. 이들은 일반 영상으로 학습하고 저속 회전익 벤치마크 [7–10]로 평가하여, 카메라가 크게 움직일 때 대상이 검색 영역을 벗어나는 문제가 드러나지 않았다. 본 논문의 기여는 (i) 표준 벤치마크와 실제 비행 영상(BioDrone [11])의 실패 원인 비교, (ii) CMC와 UAV·소형 객체 미세조정의 결합 및 시너지, (iii) 6개 추적기에 대한 CMC 일반성과 동일 미세조정 공정 비교이다.'),
  H('2. 방법'),
  P([B('CMC. '), T('직전·현재 프레임을 폭 640px로 축소하고, 대상 주변을 제외한 영역의 Shi-Tomasi 특징점 [13]을 피라미드 LK [14]로 추적하여 RANSAC [15]으로 호모그래피를 추정한다. 직전 예측 중심을 변환한 위치를 검색 영역 중심으로 쓰며, 추정이 퇴화하면 중심을 화면 안으로 제한한다.')]),
  P([B('미세조정. '), T('공식 가중치에서 BioDrone train 270개(튜닝용 30개 제외), VisDrone-SOT train 86개, LaTOT [12] 433개를 1:1:1로 20 epoch 학습한다(LR 4×10⁻⁵, 단일 GPU 23분). LaTOT는 극소형 대상(중앙값 13px) 표현을 유지하기 위해 포함하였다.')]),
  H('3. 실험'),
  P('표준 4종, BioDrone val(100개, 대상이 보이는 프레임만 채점), SatSOT [17](위성 극소형)에서 OPE로 평가하였다. 추적 실패는 IoU=0이 5프레임 이상 지속된 경우이다. FPS는 RTX PRO 6000에서 전처리를 포함한 값이다.'),
  cap('표 1. 경량 추적기 비교 (Success AUC)'),
  table(['모델', '표준4종', 'BioDrone', '+CMC', 'FPS'], [
    ['ORTrack-DeiT [1]', '64.8', '26.6', '31.3', '308'],
    ['AVTrack-D [2]', '63.5', '25.4', '33.9', '485'],
    ['SGLATrack [3]', '63.3', '24.9', '32.6', '386'],
    ['LETrack [4]', '63.7', '27.5', '34.2', '174'],
    ['OSTrack ViT-B [5]', '-', '30.9', '-', '139'],
    ['AVTrack-DeiT + UAV FT', '-', '24.2', '32.7', '322'],
    ['제안 (FT + CMC)', '63.7', '27.8', '35.7', '124'],
  ], [1689, 740, 740, 740, 740]),
  cap('표 2. Ablation (ORTrack-DeiT, BioDrone val)'),
  table(['CMC', 'FT 데이터', 'AUC', '실패(안/밖)', 'SatSOT'], [
    ['', '-', '26.6', '313 (170/143)', '38.0'],
    ['✓', '-', '31.3', '200 (107/93)', '37.4'],
    ['✓', 'UAV', '35.3', '163 (61/102)', '30.0'],
    ['✓', 'UAV+소형', '35.7', '166 (77/89)', '38.0'],
  ], [500, 1000, 700, 1549, 900]),
  img('figs/paper/fig_failure_cause.png', 280),
  cap('그림 1. 실패 원인: 검색 영역 안 혼동 / 밖 이탈'),
  P('CMC는 6개 추적기 모두에서 BioDrone AUC를 +4.7~+8.6 높였고 표준 4종에서는 손해가 없었다(64.8→64.7). 미세조정 단독은 +1.2이지만 CMC와 결합하면 +9.1로 개별 효과의 합을 넘는다. CMC가 대상을 영역 안에 유지하고, 미세조정이 영역 안 유사 물체 혼동을 줄이기 때문이다(170→77). UAV 데이터만 쓰면 SatSOT가 30.0으로 떨어지지만 LaTOT를 함께 쓰면 38.0으로 유지된다. 해상(SeaDronesSee) 영상에서도 72.7→74.2로 손해가 없다.'),
  P([B('한계. '), T('튜닝 분할에서는 UAV만 쓴 미세조정이 높았으나(43.3 vs 39.2) 소형 객체 일반화 목적에 따라 UAV+소형을 택하였다. CMC는 5.1 ms를 추가하며, 남은 실패의 54%는 카메라 회전이 클 때의 영역 이탈이다.')]),
  H('4. 결론'),
  P('실제 비행 영상에서 경량 ViT 추적기의 주요 실패 원인이 카메라 움직임에 따른 검색 영역 이탈임을 보이고, CMC와 UAV·소형 객체 미세조정으로 AUC +9.1, 실패 47% 감소를 얻었다. 향후 회전에 강한 표현 학습과 Hailo-8 NPU 배포를 연구한다.'),
  H('참고문헌'),
  ...[
    '[1] Y. Wu et al., "Learning occlusion-robust vision transformers for real-time UAV tracking," CVPR, 2025.',
    '[2] Y. Wu et al., "Learning an adaptive and view-invariant vision transformer for real-time UAV tracking," ICML, 2024.',
    '[3] "Similarity-guided layer-adaptive vision transformer for UAV tracking," CVPR, 2025.',
    '[4] "Toward low-cost yet effective temporal learning for UAV tracking," CVPR, 2026.',
    '[5] B. Ye et al., "Joint feature learning and relation modeling for tracking: A one-stream framework," ECCV, 2022.',
    '[6] H. Touvron et al., "Training data-efficient image transformers & distillation through attention," ICML, 2021.',
    '[7] M. Mueller et al., "A benchmark and simulator for UAV tracking," ECCV, 2016.',
    '[8] S. Li and D.-Y. Yeung, "Visual object tracking for unmanned aerial vehicles," AAAI, 2017.',
    '[9] D. Du et al., "The unmanned aerial vehicle benchmark," ECCV, 2018.',
    '[10] P. Zhu et al., "Detection and tracking meet drones challenge," TPAMI, 2021.',
    '[11] X. Zhao et al., "BioDrone: A bionic drone-based single object tracking benchmark," IJCV, 2024.',
    '[12] Y. Zhu et al., "Tiny object tracking: A large-scale dataset and a baseline," TNNLS, 2024.',
    '[13] J. Shi and C. Tomasi, "Good features to track," CVPR, 1994.',
    '[14] B. D. Lucas and T. Kanade, "An iterative image registration technique," IJCAI, 1981.',
    '[15] M. A. Fischler and R. C. Bolles, "Random sample consensus," Commun. ACM, 1981.',
    '[16] N. Aharon et al., "BoT-SORT," arXiv:2206.14651, 2022.',
    '[17] "SatSOT: A benchmark dataset for satellite video single object tracking," TGRS, 2022.',
  ].map((r) => P(r, { size: 7, after: 10, align: AlignmentType.LEFT })),
];

const page = { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } };
const doc = new Document({
  sections: [
    { properties: { page }, children: head },
    { properties: { page, type: SectionType.CONTINUOUS, column: { count: 2, space: 340 } }, children: body },
  ],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(path.join(ROOT, 'paper/draft_icros2027.docx'), b); console.log('saved'); });

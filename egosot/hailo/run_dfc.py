"""ORTrack(드론 FT) ONNX → Hailo-8 HEF 변환 (DFC 설치 후 실행).

준비물 (모두 준비됨):
  hailo/onnx/ortrack_uavft_slim.onnx   (tools/export_onnx.py + onnxslim, PyTorch 대비 오차 < 5e-6)
  hailo/calib_uav_1024.npz             (tools/make_hailo_calib.py, 학습 분할에서만 1024쌍, uint8 NHWC)

설치 (사용자가 Hailo Developer Zone 에서 받은 whl 을 hailo/ 에 두고):
  pip install --target /home/airlab/user/kdh/SOT/hailo_deps hailo/hailo_dataflow_compiler-*.whl
실행:
  PYTHONPATH=/home/airlab/user/kdh/SOT/hailo_deps python hailo/run_dfc.py --level 2   # 빠른 확인
  PYTHONPATH=/home/airlab/user/kdh/SOT/hailo_deps python hailo/run_dfc.py --level 4   # 최종 (GPU 필요, 수 시간)

이전 OSTrack 시도(hailo-drone-mot, docs/ostrack_hailo_probe.md) 교훈:
  - CE(후보 제거) 모듈은 파서 실패 → ORTrack 은 CE 없음 (CE_LOC=[])
  - 1024장 미만이면 일부 단계가 64장만 사용 → calibset_size 명시
  - conv2 reshape / attention matmul 분할 오류 → 공식 ViT alls(zp_comp_block 등) 사용
"""
import argparse, os
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--onnx', default='hailo/onnx/ortrack_final_slim.onnx')
p.add_argument('--calib', default='hailo/calib_uav_1024.npz')
p.add_argument('--level', type=int, default=2)
p.add_argument('--out', default='hailo/out')
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

from hailo_sdk_client import ClientRunner  # noqa: E402

runner = ClientRunner(hw_arch='hailo8')
runner.translate_onnx_model(a.onnx, 'ortrack_uavft',
                            start_node_names=['template', 'search'],
                            end_node_names=['score_map', 'size_map', 'offset_map'],
                            net_input_shapes={'template': [1, 3, 128, 128], 'search': [1, 3, 256, 256]})
runner.save_har(f'{a.out}/ortrack_uavft_parsed.har')
print('parse OK')

mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
inputs = runner.get_hn_dict()['net_params']['input_layers_names'] if 'net_params' in runner.get_hn_dict() else None
alls = f"""
norm_t = normalization({mean}, {std}, ortrack_uavft/input_layer1)
norm_s = normalization({mean}, {std}, ortrack_uavft/input_layer2)
model_optimization_config(calibration, batch_size=1, calibset_size=1024)
model_optimization_flavor(optimization_level={a.level}, compression_level=0)
pre_quantization_optimization(equalization, policy=enabled)
"""
runner.load_model_script(alls)
c = np.load(a.calib)
# 입력 레이어 이름은 파싱 결과 순서(template, search)에 맞춰 매핑 — 다르면 runner.get_hn_dict() 로 확인
calib = {'ortrack_uavft/input_layer1': c['template'].astype(np.float32),
         'ortrack_uavft/input_layer2': c['search'].astype(np.float32)}
runner.optimize(calib)
runner.save_har(f'{a.out}/ortrack_uavft_quant_l{a.level}.har')
print('optimize OK')

hef = runner.compile()
with open(f'{a.out}/ortrack_uavft_l{a.level}.hef', 'wb') as f:
    f.write(hef)
print('HEF saved', f'{a.out}/ortrack_uavft_l{a.level}.hef')

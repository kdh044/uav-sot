"""Ego-motion-aware 추적 보조 함수 (실험용, env 로만 켜짐).

M2: 누적 카메라 회전만큼 검색 영역을 역회전해 잘라 템플릿 방향과 정렬.
"""
import math
import cv2
import numpy as np


def homography_rotation(Hm):
    """이전→현재 호모그래피의 회전 성분 (rad, 이미지 좌표계 y-down 기준, 반시계 화면 회전이 음수)."""
    return math.atan2(Hm[1, 0], Hm[0, 0])


def rotated_crop(image, box, factor, out_sz, angle):
    """box(x,y,w,h, 정렬 프레임 크기) 중심으로 한 변 sqrt(w*h)*factor 영역을 angle(rad) 만큼 역회전해 out_sz 로 자름.

    반환: patch(out_sz,out_sz,3), resize_factor, att_mask(1=패딩), M(원본→patch 2x3 affine)
    """
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    crop_sz = math.ceil(math.sqrt(w * h) * factor)
    if crop_sz < 1:
        raise Exception('Too small bounding box.')
    scale = out_sz / crop_sz
    # 카메라가 이미지 내용을 angle 만큼 돌렸으므로, 같은 각도로 getRotationMatrix2D 를 적용하면 상쇄됨
    M = cv2.getRotationMatrix2D((cx, cy), math.degrees(angle), scale)
    M[0, 2] += out_sz / 2 - cx
    M[1, 2] += out_sz / 2 - cy
    patch = cv2.warpAffine(image, M, (out_sz, out_sz), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    valid = cv2.warpAffine(np.ones(image.shape[:2], np.uint8), M, (out_sz, out_sz),
                           flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    att_mask = (1 - valid).astype(np.float64)
    return patch, scale, att_mask, M


def map_back_rotated(pred_box_crop_px, M, angle):
    """patch 픽셀 좌표의 (cx,cy,w,h) → 원본 좌표 정렬 프레임 박스(cx,cy,w,h)와 축정렬 외접 박스(x,y,w,h)."""
    cx, cy, w, h = pred_box_crop_px
    Minv = cv2.invertAffineTransform(M)
    ox, oy = Minv @ np.array([cx, cy, 1.0])
    s = math.hypot(Minv[0, 0], Minv[0, 1])
    w0, h0 = w * s, h * s
    c, sn = abs(math.cos(angle)), abs(math.sin(angle))
    aw, ah = w0 * c + h0 * sn, w0 * sn + h0 * c
    return (ox, oy, w0, h0), [ox - aw / 2, oy - ah / 2, aw, ah]

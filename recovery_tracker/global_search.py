"""Prior-independent first-frame patch proposals."""
import cv2
import numpy as np


def overlap(a, b):
    lo = np.maximum(a[..., :2], b[..., :2])
    hi = np.minimum(a[..., :2] + a[..., 2:], b[..., :2] + b[..., 2:])
    intersection = np.maximum(hi - lo, 0).prod(-1)
    return intersection / np.maximum(a[..., 2:].prod(-1) + b[..., 2:].prod(-1) - intersection, 1e-9)


def make_template(image, box):
    x, y, w, h = box
    shape = max(4, round(2*w)), max(4, round(2*h))
    return cv2.getRectSubPix(image, shape, (x+w/2-.5,y+h/2-.5)), np.array([w,h])


def search(image, template, size):
    size = np.asarray(size)
    candidates = []
    for ratio in (.7, 1., 1.4):
        width = max(1, round(template.shape[1] * ratio))
        height = max(1, round(template.shape[0] * ratio))
        if width > image.shape[1] or height > image.shape[0]:
            continue
        patch = cv2.resize(template, (width, height))
        score = cv2.matchTemplate(image, patch, cv2.TM_CCOEFF_NORMED)
        peaks = (score == cv2.dilate(score, np.ones((9,9),np.uint8)))
        ids = np.flatnonzero(peaks)
        ids = ids[np.argsort(score.flat[ids])[-8:]]
        for j in ids:
            y,x = divmod(int(j),score.shape[1])
            w,h = size*ratio
            candidates.append((float(score[y,x]),[x+(patch.shape[1]-w)/2,y+(patch.shape[0]-h)/2,w,h]))
    ranked = []
    for _, box in sorted(candidates, key=lambda item:item[0], reverse=True):
        if ranked and overlap(np.array(box),np.array(ranked)).max()>.3:
            continue
        ranked.append(box)
        if len(ranked)==8:
            break
    return np.asarray(ranked, dtype=float).reshape(-1, 4)




def patch_scores(image, template, boxes):
    """Compare both proposal sources using the same contextual appearance score."""
    scores = np.full(len(boxes), -np.inf)
    height, width = image.shape
    for i, (x, y, w, h) in enumerate(boxes):
        cx, cy = x+w/2, y+h/2
        if not (0 <= cx < width and 0 <= cy < height and 1 <= w <= width and 1 <= h <= height):
            continue
        patch = cv2.getRectSubPix(image, (max(4,round(2*w)),max(4,round(2*h))),
                                 (float(cx-.5),float(cy-.5)))
        patch = cv2.resize(patch, (template.shape[1],template.shape[0]))
        a = patch.astype(np.float32) - patch.mean()
        b = template.astype(np.float32) - template.mean()
        norm = np.linalg.norm(a)*np.linalg.norm(b)
        if norm > 0:
            scores[i] = np.sum(a*b)/norm
    return scores

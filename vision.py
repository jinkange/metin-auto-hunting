
# vision.py
# 창 스크린샷(mss) + 템플릿 매칭(OpenCV)

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from mss import mss

from config import (
    COORD_REGION_REL, COORD_BIN_THRESH, DIGIT_MATCH_THRESH,
    TH_CHAR, TH_MONSTER, TH_SPECIAL,
    DEBUG_SAVE_FAIL_COORD, DEBUG_DIR
)

ASSET = Path(__file__).parent / "assets"


def _imread_gray(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지 로드 실패: {p}")
    return img


def _load_digit_templates():
    # 지원:
    #  - assets/digits/0.png .. 9.png
    #  - assets/digits/0/*.png .. 9/*.png
    out = {}
    ddir = ASSET / "digits"
    for d in range(10):
        tpls = []
        many_dir = ddir / str(d)
        if many_dir.exists() and many_dir.is_dir():
            for p in sorted(many_dir.glob("*.png")):
                tpls.append(_imread_gray(p))
        single = ddir / f"{d}.png"
        if single.exists():
            tpls.append(_imread_gray(single))
        if not tpls:
            raise FileNotFoundError(f"digits 템플릿 없음: {many_dir} 또는 {single}")
        out[d] = tpls
    return out


DIGIT_TPL = _load_digit_templates()


def load_templates(subdir: str):
    dirp = ASSET / subdir
    if not dirp.exists():
        return []
    return [_imread_gray(p) for p in sorted(dirp.glob("*.png"))]


CHAR_TPL = load_templates("character")
MON_TPL = load_templates("monster")
SPECIAL_TPL = [
    load_templates("special/1"),
    load_templates("special/2"),
    load_templates("special/3"),
]


class Grabber:
    def __init__(self, rect):
        self.rect = rect  # (l,t,r,b)
        self.sct = mss()

    def grab_gray(self):
        l, t, r, b = self.rect
        mon = {"left": l, "top": t, "width": r - l, "height": b - t}
        img = np.array(self.sct.grab(mon))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    def grab_region_gray_rel(self, rel_region):
        l, t, _, _ = self.rect
        x, y, w, h = rel_region
        mon = {"left": l + x, "top": t + y, "width": w, "height": h}
        img = np.array(self.sct.grab(mon))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)


def _binarize(gray):
    _, bw = cv2.threshold(gray, COORD_BIN_THRESH, 255, cv2.THRESH_BINARY_INV)
    return cv2.medianBlur(bw, 3)


def _connected_components(bw):
    num, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    boxes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 18:
            continue
        if h < 6 or w < 3:
            continue
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])
    return boxes


def _classify_digit(roi):
    ref = DIGIT_TPL[0][0]
    th, tw = ref.shape[:2]
    roi_rs = cv2.resize(roi, (tw, th), interpolation=cv2.INTER_AREA)

    best_d, best_s = None, -1.0
    for d, tpls in DIGIT_TPL.items():
        for tpl in tpls:
            s = float(cv2.matchTemplate(roi_rs, tpl, cv2.TM_CCOEFF_NORMED).max())
            if s > best_s:
                best_s, best_d = s, d

    if best_s < DIGIT_MATCH_THRESH:
        return None
    return best_d


def read_game_coord(grabber: Grabber):
    gray = grabber.grab_region_gray_rel(COORD_REGION_REL)
    bw = _binarize(gray)
    boxes = _connected_components(bw)

    digits = []
    for (x, y, w, h) in boxes:
        roi = bw[y : y + h, x : x + w]
        d = _classify_digit(roi)
        if d is None:
            continue
        digits.append((x + w / 2.0, d))

    if len(digits) < 2:
        if DEBUG_SAVE_FAIL_COORD:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(DEBUG_DIR, f"coord_fail_{int(np.random.rand()*1e9)}.png"), gray)
        return None

    digits.sort(key=lambda t: t[0])
    xs = [t[0] for t in digits]
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    if not gaps:
        return None

    split_i = int(np.argmax(gaps))
    left = digits[: split_i + 1]
    right = digits[split_i + 1 :]
    if not left or not right:
        return None

    gx = int("".join(str(d) for _, d in left))
    gy = int("".join(str(d) for _, d in right))
    return (gx, gy)


def _match_all(gray, templates, th):
    found = []
    if not templates:
        return found
    for tpl in templates:
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= th)
        h, w = tpl.shape[:2]
        for x, y in zip(xs, ys):
            found.append((int(x), int(y), w, h))
    return found


def find_character_center(window_gray):
    pts = _match_all(window_gray, CHAR_TPL, TH_CHAR)
    if not pts:
        return None
    xs = [x + w // 2 for x, y, w, h in pts]
    ys = [y + h // 2 for x, y, w, h in pts]
    return (sum(xs) // len(xs), sum(ys) // len(ys))


def find_monsters(window_gray):
    boxes = _match_all(window_gray, MON_TPL, TH_MONSTER)
    return [(x + w // 2, y + h // 2) for x, y, w, h in boxes]


def check_special(window_gray):
    present = []
    for tpls in SPECIAL_TPL:
        present.append(len(_match_all(window_gray, tpls, TH_SPECIAL)) > 0)
    return present

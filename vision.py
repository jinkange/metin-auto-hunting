# vision.py
# - main.py에서 요구하는 vision.Grabber 제공
# - 윈도우 창(또는 주어진 rect) 기준으로 스크린샷 캡처
# - 좌표(예: "23x 25", "24x24") 숫자 템플릿 기반 파싱 유틸 포함
#
# 필요 패키지: opencv-python, numpy, pillow (pyautogui 사용 시 pyautogui)
#
# 사용 예)
#   grabber = vision.Grabber(rect)              # rect: (l,t,r,b) 또는 (l,t,w,h)
#   img = grabber.grab_gray_abs()
#   coord = vision.read_game_coord(grabber, region=(x,y,w,h), debug=True)

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import cv2

try:
    from PIL import ImageGrab  # pillow
except Exception:
    ImageGrab = None

# (선택) pyautogui fallback
try:
    import pyautogui
except Exception:
    pyautogui = None


print(f"vision loaded from: {__file__}")


# -----------------------------
# Unicode 경로 안전 imread (한글 경로 OpenCV imread 실패 방지)
# -----------------------------
def _imread_gray_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return img


def _load_digit_templates(digits_dir="assets/digits"):
    """
    digits 템플릿 로딩
    지원 형태:
      assets/digits/0.png ... 9.png (단일)
      assets/digits/0/*.png ... (복수)
    """
    base = Path(digits_dir)
    if not base.exists():
        raise FileNotFoundError(f"[vision] digits 폴더가 없습니다: {base.resolve()}")

    templates = {i: [] for i in range(10)}

    # 1) 단일 파일 0.png~9.png
    single_found = False
    for i in range(10):
        p = base / f"{i}.png"
        if p.exists():
            img = _imread_gray_unicode(p)
            if img is None:
                raise RuntimeError(f"[vision] digits 템플릿 로딩 실패: {p}")
            templates[i].append(img)
            single_found = True

    # 2) 폴더별 복수 템플릿 0/,1/...
    for i in range(10):
        d = base / str(i)
        if d.exists() and d.is_dir():
            for p in sorted(d.glob("*.png")):
                img = _imread_gray_unicode(p)
                if img is not None:
                    templates[i].append(img)

    if not any(len(v) for v in templates.values()):
        raise FileNotFoundError(
            f"[vision] digits 템플릿이 비어있습니다.\n"
            f"  - {base}/0.png..9.png 또는\n"
            f"  - {base}/0/*.png..{base}/9/*.png\n"
            f"중 하나로 넣어주세요."
        )

    return templates


# -----------------------------
# Grabber: 윈도우/화면 캡처 유틸
# -----------------------------
class Grabber:
    """
    rect는 아래 2가지 모두 지원:
      - (left, top, right, bottom)
      - (left, top, width, height)
    """
    def __init__(self, rect):
        if rect is None or len(rect) != 4:
            raise ValueError("[vision.Grabber] rect는 (l,t,r,b) 또는 (l,t,w,h) 4튜플이어야 합니다.")

        a, b, c, d = rect
        # (l,t,r,b)인지 (l,t,w,h)인지 판별: r>b? w/h? 애매하니 안전하게 계산
        # 1) right/bottom처럼 보이면 width/height 계산
        if c > a and d > b and (c - a) > 30 and (d - b) > 30:
            # 대부분 (l,t,r,b)
            self.left, self.top = int(a), int(b)
            self.width, self.height = int(c - a), int(d - b)
        else:
            # (l,t,w,h)로 간주
            self.left, self.top = int(a), int(b)
            self.width, self.height = int(c), int(d)

        self.left = max(0, self.left)
        self.top = max(0, self.top)
        self.width = max(1, self.width)
        self.height = max(1, self.height)

    def _grab_rgb(self, region_abs):
        """
        region_abs: (l,t,w,h) 절대 좌표
        return: RGB np.ndarray
        """
        l, t, w, h = region_abs
        if ImageGrab is not None:
            img = ImageGrab.grab(bbox=(l, t, l + w, t + h))
            arr = np.array(img)  # RGB
            return arr
        if pyautogui is not None:
            img = pyautogui.screenshot(region=(l, t, w, h))
            arr = np.array(img)  # RGB
            return arr
        raise RuntimeError("[vision] ImageGrab/pyautogui 둘 다 사용 불가: pillow 또는 pyautogui 설치 필요")

    def grab_bgr_abs(self):
        """창 전체를 BGR(OpenCV)로 캡처"""
        rgb = self._grab_rgb((self.left, self.top, self.width, self.height))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    def grab_gray_abs(self):
        """창 전체를 GRAY로 캡처"""
        bgr = self.grab_bgr_abs()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def grab_bgr_rel(self, region):
        """창 기준 상대좌표 region=(x,y,w,h)만 BGR 캡처"""
        x, y, w, h = map(int, region)
        rgb = self._grab_rgb((self.left + x, self.top + y, w, h))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def grab_gray_rel(self, region):
        """창 기준 상대좌표 region=(x,y,w,h)만 GRAY 캡처"""
        bgr = self.grab_bgr_rel(region)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# -----------------------------
# 좌표(예: "23x 25") 읽기 - 숫자 템플릿 기반
# -----------------------------
_DIGITS = None

def _ensure_digits_loaded():
    global _DIGITS
    if _DIGITS is None:
        _DIGITS = _load_digit_templates("assets/digits")
    return _DIGITS


def _binarize(gray, thresh=160):
    # 숫자 글자가 밝고 배경이 어두운/밝은 경우 대응: INV 사용
    # 상황에 따라 thresh 조절 필요
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    bw = cv2.medianBlur(bw, 3)
    return bw


def _connected_components(bw):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    boxes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 15:
            continue
        # 너무 큰 덩어리(배경 전체)가 잡히면 제외
        if w > bw.shape[1] * 0.9 and h > bw.shape[0] * 0.9:
            continue
        boxes.append((x, y, w, h, area))
    boxes.sort(key=lambda b: b[0])
    return boxes


def _classify_digit(roi_bw, digit_templates, match_thresh=0.55):
    # 템플릿 기준으로 리사이즈 후 최고점 선택
    # 템플릿이 여러 장일 수 있으므로 max over all
    best_d, best_s = None, -1.0

    # 기준 크기: 가장 많이 쓰는 0번 첫 템플릿
    ref = digit_templates[0][0]
    th, tw = ref.shape[:2]
    roi_rs = cv2.resize(roi_bw, (tw, th), interpolation=cv2.INTER_AREA)

    for d in range(10):
        for tpl in digit_templates[d]:
            tpl_rs = tpl
            if tpl.shape[:2] != (th, tw):
                tpl_rs = cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(roi_rs, tpl_rs, cv2.TM_CCOEFF_NORMED)
            s = float(res.max())
            if s > best_s:
                best_s, best_d = s, d

    if best_s < match_thresh:
        return None, best_s
    return best_d, best_s


def read_game_coord(grabber: Grabber, region, *,
                   thresh=160, match_thresh=0.55,
                   scale=2.0, debug=False):
    """
    게임 우측상단의 좌표 문자열(예: '23x 25', '24x24')에서 (x,y) 반환.
    region: (x,y,w,h) - 창 기준 상대 좌표

    - 숫자만 인식하고, x/공백 등은 무시
    - 숫자 덩어리를 좌/우로 나누기 위해 "가장 큰 간격(gap)"으로 split
    """
    digits_tpl = _ensure_digits_loaded()

    gray = grabber.grab_gray_rel(region)

    # 작은 글씨면 확대가 도움이 됨
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    bw = _binarize(gray, thresh=thresh)
    boxes = _connected_components(bw)

    picks = []  # (x_center, digit, score)
    for (x, y, w, h, area) in boxes:
        # 너무 얇거나 납작한 것은 x 문자 가능성이 높음 -> 일단 분류로 걸러짐
        roi = bw[y:y+h, x:x+w]
        d, s = _classify_digit(roi, digits_tpl, match_thresh=match_thresh)
        if d is None:
            continue
        picks.append((x + w/2.0, d, s))

    if debug:
        print(f"[COORD] region={region}, scale={scale}, boxes={len(boxes)}, digits={len(picks)}")

    if len(picks) < 2:
        if debug:
            print("[COORD] FAIL digits < 2")
        return None

    picks.sort(key=lambda t: t[0])
    xs = [p[0] for p in picks]
    gaps = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
    if not gaps:
        if debug:
            print("[COORD] FAIL no gaps")
        return None

    split_i = int(np.argmax(gaps))
    left = picks[:split_i+1]
    right = picks[split_i+1:]

    if not left or not right:
        if debug:
            print("[COORD] FAIL split empty")
        return None

    def to_int(group):
        return int("".join(str(p[1]) for p in group))

    try:
        gx = to_int(left)
        gy = to_int(right)
        if debug:
            print(f"[COORD] OK -> ({gx}, {gy}) | left={''.join(str(p[1]) for p in left)} right={''.join(str(p[1]) for p in right)}")
        return (gx, gy)
    except Exception as e:
        if debug:
            print(f"[COORD] FAIL parse: {e}")
        return None


# -----------------------------
# (선택) 아래 함수들은 기존 코드가 호출할 수 있어 남겨둠
# - 필요 없으면 삭제해도 됨
# -----------------------------
def match_template_points(gray, template_gray, threshold=0.8):
    res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= threshold)
    pts = list(zip(xs, ys))
    return pts
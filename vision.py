# vision.py
import cv2
import numpy as np
import pyautogui
from pathlib import Path
from config import COORD_REGION, COORD_BIN_THRESH, DIGIT_MATCH_THRESH

ASSET = Path("assets")

def screenshot(region=None):
    img = pyautogui.screenshot(region=region)
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return gray

def _load_digit_templates():
    digits = {}
    for i in range(10):
        p = ASSET / "digits" / f"{i}.png"
        if not p.exists():
            raise FileNotFoundError(f"digits 템플릿이 없습니다: {p}")
        digits[i] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return digits

DIGIT_TPL = _load_digit_templates()

def _binarize(gray):
    # 글자가 밝고 배경이 어두운/밝은 경우가 있어 OTSU도 가능하지만, 우선 고정값 제공
    _, bw = cv2.threshold(gray, COORD_BIN_THRESH, 255, cv2.THRESH_BINARY_INV)
    # 작은 노이즈 제거
    bw = cv2.medianBlur(bw, 3)
    return bw

def _connected_components(bw):
    # 숫자 후보 덩어리(연결요소) 찾기
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    boxes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 20:
            continue
        # 너무 납작/너무 긴 것(구분자 등)은 일단 후보로 두되, 나중에 점수로 걸러짐
        boxes.append((x, y, w, h, area))
    # 왼쪽→오른쪽 정렬
    boxes.sort(key=lambda b: b[0])
    return boxes

def _classify_digit(roi):
    # ROI를 각 템플릿 크기에 맞춰 점수 계산 → 최고 점수 digit 선택
    best_digit = None
    best_score = -1.0

    # 템플릿 크기 기준으로 ROI 리사이즈(템플릿이 모두 같은 크기일 때 가장 안정적)
    # 여기서는 0 템플릿 크기 기준 사용
    ref = DIGIT_TPL[0]
    th, tw = ref.shape[:2]
    roi_rs = cv2.resize(roi, (tw, th), interpolation=cv2.INTER_AREA)

    for d, tpl in DIGIT_TPL.items():
        res = cv2.matchTemplate(roi_rs, tpl, cv2.TM_CCOEFF_NORMED)
        score = float(res.max())
        if score > best_score:
            best_score = score
            best_digit = d

    if best_score < DIGIT_MATCH_THRESH:
        return None, best_score
    return best_digit, best_score

def read_game_coord():
    """
    우측상단 좌표(예: '23x 25')를 (x, y) int로 반환.
    실패 시 None 반환.
    """
    gray = screenshot(region=COORD_REGION)
    bw = _binarize(gray)
    boxes = _connected_components(bw)

    # 각 박스에서 digit 분류
    digits = []  # (x_center, digit)
    for (x, y, w, h, area) in boxes:
        roi = bw[y:y+h, x:x+w]
        d, score = _classify_digit(roi)
        if d is None:
            continue
        digits.append((x + w/2.0, d))

    if len(digits) < 2:
        return None

    # 좌표의 좌/우 숫자 그룹을 나누기:
    # - x_center 기준 정렬 후, "가장 큰 간격"을 분할점으로 사용
    digits.sort(key=lambda t: t[0])
    xs = [t[0] for t in digits]
    gaps = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
    if not gaps:
        return None
    split_i = int(np.argmax(gaps))  # 가장 큰 gap 인덱스

    left = digits[:split_i+1]
    right = digits[split_i+1:]

    if not left or not right:
        return None

    def to_int(group):
        # 그룹 내 숫자를 순서대로 이어붙임
        s = "".join(str(d) for _, d in group)
        return int(s)

    try:
        gx = to_int(left)
        gy = to_int(right)
        return (gx, gy)
    except:
        return None
import cv2
import numpy as np
import pyautogui
from pathlib import Path

from config import COORD_REGION, MAP_W, MAP_H

ASSET = Path("assets")

# ----------------------------
# 1) 템플릿 로딩 (단일 or 다중 템플릿 모두 지원)
# - 권장: assets/digits/0.png ~ 9.png (단일)
# - 더 권장: assets/digits/0/*.png 처럼 다중 템플릿
# ----------------------------
def _load_digit_templates():
    digits = {i: [] for i in range(10)}

    # 다중 폴더 구조 우선: digits/0/*.png
    multi_ok = (ASSET / "digits" / "0").exists()
    if multi_ok:
        for i in range(10):
            ddir = ASSET / "digits" / str(i)
            for p in ddir.glob("*.png"):
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    digits[i].append(img)
    else:
        for i in range(10):
            p = ASSET / "digits" / f"{i}.png"
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"digits 템플릿이 없습니다: {p}")
            digits[i].append(img)

    return digits

DIGIT_TPL = _load_digit_templates()

def screenshot_gray(region=None):
    img = pyautogui.screenshot(region=region)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

# ----------------------------
# 2) 전처리(핵심): 배경 변화(밤/낮)에 강하게
# - CLAHE로 대비 보정
# - OTSU로 자동 임계값
# - morphology로 노이즈 제거/획 보강
# ----------------------------
def preprocess(gray):
    # 대비 보정(배경이 어둡거나 밝아도 숫자 대비를 끌어올림)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    # 가우시안 블러 약하게(노이즈 완화)
    g = cv2.GaussianBlur(g, (3, 3), 0)

    # OTSU 자동 이진화 (숫자가 밝을 수도/어두울 수도 있어 두 방향 다 시도)
    _, bw1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, bw2 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 둘 중 “연결요소 개수/면적”이 숫자 분리에 더 유리한 걸 선택
    def score_bw(bw):
        n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
        # 너무 많으면(노이즈) 점수 낮게, 너무 적어도 낮게
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n)]
        if not areas:
            return -999
        a = np.array(areas)
        # 작은 노이즈 제거 후 개수와 면적 분포로 점수
        good = a[a > 20]
        return len(good) - 0.02 * len(a)

    bw = bw1 if score_bw(bw1) >= score_bw(bw2) else bw2

    # morphology로 글자 획 보강/노이즈 제거
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

    return g, bw

# ----------------------------
# 3) 숫자 후보 분리: contour 기반이 연결요소보다 안정적일 때가 많음
# ----------------------------
def extract_candidates(bw):
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = bw.shape[:2]

    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh

        # 필터: 너무 작으면 노이즈
        if area < 40:
            continue
        # 너무 큰 덩어리(좌표 패널 전체 등) 제거
        if ww > w * 0.9 or hh > h * 0.9:
            continue

        # 숫자는 보통 세로가 어느 정도 있음(경험적)
        if hh < 8:
            continue

        boxes.append((x, y, ww, hh))

    # 왼->오 정렬
    boxes.sort(key=lambda b: b[0])
    return boxes

# ----------------------------
# 4) 숫자 분류: 다중 템플릿 + 크기 정규화
# ----------------------------
def classify_digit(roi):
    # 템플릿 기준 크기로 리사이즈하여 matchTemplate 점수 최대를 사용
    best_d, best_s = None, -1.0

    # 기준 크기: 각 숫자의 첫 템플릿(0번) 기준
    ref = DIGIT_TPL[0][0]
    th, tw = ref.shape[:2]

    roi_rs = cv2.resize(roi, (tw, th), interpolation=cv2.INTER_AREA)

    for d in range(10):
        for tpl in DIGIT_TPL[d]:
            # 템플릿도 동일 크기 아니면 맞춰줌
            if tpl.shape != (th, tw):
                tpl_rs = cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_AREA)
            else:
                tpl_rs = tpl
            res = cv2.matchTemplate(roi_rs, tpl_rs, cv2.TM_CCOEFF_NORMED)
            s = float(res.max())
            if s > best_s:
                best_s, best_d = s, d

    return best_d, best_s

# ----------------------------
# 5) 좌표 파싱: "24x25"에서 x 구분이 애매할 수 있음
#    -> 분할 후보들을 전부 만들어 (맵 범위, 이전좌표, 이동량)로 최적 선택
# ----------------------------
def split_candidates(digits):
    """
    digits: [(x_center, digit, score), ...]  좌->우 정렬
    가능한 모든 분할 i(왼쪽 1개 이상, 오른쪽 1개 이상)를 후보로 생성
    """
    cands = []
    n = len(digits)
    for i in range(1, n):
        left = digits[:i]
        right = digits[i:]
        lx = int("".join(str(d) for _, d, _ in left))
        ry = int("".join(str(d) for _, d, _ in right))
        # 후보 점수: 평균 매칭 점수
        s = (sum(sc for _, _, sc in left) / len(left) + sum(sc for _, _, sc in right) / len(right)) / 2
        cands.append(((lx, ry), s, i))
    return cands

def choose_best_coord(cands, last_coord=None, max_step=5):
    """
    후보 중 가장 그럴듯한 (x,y) 선택:
    1) 맵 범위 (0..MAP_W-1, 0..MAP_H-1)
    2) last_coord가 있으면 이동량(체비셰프) <= max_step*2 정도 우선 (게임 딜레이/오차 고려)
    3) 후보 점수(템플릿 매칭 점수) 우선
    """
    best = None
    best_val = -1e9

    for (xy, s, split_i) in cands:
        x, y = xy
        if not (0 <= x < MAP_W and 0 <= y < MAP_H):
            continue

        bonus = 0.0
        if last_coord is not None:
            dx = abs(x - last_coord[0])
            dy = abs(y - last_coord[1])
            d = max(dx, dy)
            # 너무 튀는 좌표는 감점
            if d > max_step * 3:
                bonus -= 1.0
            else:
                bonus += 0.4

        val = s + bonus
        if val > best_val:
            best_val = val
            best = xy

    return best

# ----------------------------
# 6) 메인: 좌표 읽기(견고 버전)
# ----------------------------
_LAST_GOOD = None

def read_game_coord_robust(last_coord=None, max_step=5):
    global _LAST_GOOD
    gray = screenshot_gray(region=COORD_REGION)
    _, bw = preprocess(gray)

    boxes = extract_candidates(bw)
    if len(boxes) < 2:
        # 실패: 마지막 값 반환(있으면)
        return last_coord or _LAST_GOOD

    items = []
    for (x, y, w, h) in boxes:
        roi = bw[y:y+h, x:x+w]

        # ROI 패딩(가끔 숫자 가장자리가 잘리면 점수 급락)
        roi = cv2.copyMakeBorder(roi, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)

        d, s = classify_digit(roi)
        # 점수 컷(너무 낮으면 숫자 아님)
        if s < 0.45:
            continue

        items.append((x + w/2.0, d, s))

    if len(items) < 2:
        return last_coord or _LAST_GOOD

    items.sort(key=lambda t: t[0])

    # 분할 후보 생성 후 최적 선택
    cands = split_candidates(items)
    best = choose_best_coord(cands, last_coord=last_coord, max_step=max_step)

    if best is not None:
        _LAST_GOOD = best
        return best

    return last_coord or _LAST_GOOD

# ----------------------------
# 7) 디버그: 이미지 저장 + 박스 표시
# ----------------------------
def read_game_coord_debug(save_prefix="debug"):
    gray = screenshot_gray(region=COORD_REGION)
    g, bw = preprocess(gray)
    boxes = extract_candidates(bw)

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    items = []
    for (x, y, w, h) in boxes:
        roi = bw[y:y+h, x:x+w]
        roi = cv2.copyMakeBorder(roi, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        d, s = classify_digit(roi)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0,255,0), 1)
        cv2.putText(vis, f"{d}:{s:.2f}", (x, max(12, y-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        items.append((x + w/2.0, d, s))

    cv2.imwrite(save_prefix + "_raw.png", gray)
    cv2.imwrite(save_prefix + "_pre.png", g)
    cv2.imwrite(save_prefix + "_bw.png", bw)
    cv2.imwrite(save_prefix + "_boxes.png", vis)

    if len(items) < 2:
        return None

    items.sort(key=lambda t: t[0])
    cands = split_candidates(items)
    best = choose_best_coord(cands, last_coord=_LAST_GOOD, max_step=5)
    return best
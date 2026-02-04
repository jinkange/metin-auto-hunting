
# config.py

WINDOW_TITLE = "Metin1"

MAP_W, MAP_H = 255, 255

MOVE_STEP_DEFAULT = 1      # 1~5 (실행 중 step 명령으로 변경 가능)
MOVE_PIXEL_PER_TILE = 32   # 타일 1칸당 클릭 픽셀 이동량(게임에 맞게 조정)

NEAR_RADIUS = 3
TICK = 0.12

# 우측상단 좌표 영역(게임창 기준 상대좌표) (x, y, w, h)
# 696 42 751 57
# COORD_REGION = (696, 42, 55, 15)
COORD_REGION = (696, 42, 100, 55)

COORD_BIN_THRESH = 160
DIGIT_MATCH_THRESH = 0.55

MOVE_CHECK_TIMEOUT = 1.2
MOVE_CHECK_POLL = 0.10


# 금지 좌표들 (게임 내부 좌표)
FORBIDDEN_POINTS = [
    (10, 10),
    (120, 80),
    # ...
]

# 금지 좌표 기준 몇 칸까지 접근 금지 (요청: 2칸)
FORBIDDEN_PAD = 2

# 대각 포함(체비셰프 거리)로 금지 반경 계산할지
FORBIDDEN_USE_CHEBYSHEV = True

TH_CHAR = 0.75
TH_MONSTER = 0.78
TH_SPECIAL = 0.85

DEBUG_SAVE_FAIL_COORD = True
DEBUG_DIR = "debug"


# 좌표 전용 전처리 (밤 보정 사용 안 함)
COORD_USE_ADAPTIVE = True      # True면 adaptive threshold 사용(배경 변화에 강함)
COORD_BIN_THRESH = 160         # COORD_USE_ADAPTIVE=False일 때만 사용

# 디버깅: 좌표 인식 실패 시 이미지 저장
COORD_DEBUG_SAVE = True
COORD_DEBUG_DIR = "debug/coord_fail"

# (선택) 몬스터/이미지 인식용 밤 보정 옵션
VISION_NIGHT_MODE = False      # 몬스터 인식에만 적용할 예정


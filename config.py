# config.py
WINDOW_TITLE = "내게임창제목"

MAP_W, MAP_H = 255, 255

MOVE_STEP = 1              # 1~5
NEAR_RADIUS = 3
TICK = 0.12

# 우측상단 좌표 영역(화면 픽셀 기준)  (left, top, width, height)
# ※ 게임 창 위치가 바뀌면 같이 바뀝니다. 가능하면 "고정 위치"에서 테스트하세요.
COORD_REGION = (1500, 30, 220, 60)

# 금지 좌표(게임 내부 좌표) & 주변 2칸 접근 금지
FORBIDDEN_POINTS = [(10, 10), (120, 80)]
FORBIDDEN_PAD = 2
FORBIDDEN_USE_CHEBYSHEV = True

# 좌표 판독 파라미터(필요 시 튜닝)
COORD_BIN_THRESH = 160          # 이진화 임계값(배경/글자 밝기에 따라 조정)
DIGIT_MATCH_THRESH = 0.55       # 템플릿 매칭 점수 임계값(낮추면 오탐 늘고, 높이면 인식 실패 늘어남)

# 이동 판정
MOVE_CHECK_TIMEOUT = 1.2        # 이동 후 좌표가 바뀌는지 기다리는 최대 시간(초)
MOVE_CHECK_POLL = 0.10          # 좌표 재확인 주기(초)
# main.py
import time, keyboard
from config import *
import vision
import combat
import navigation
from input_ctrl import right_click, press_key, move_dir

state = combat.CombatState()
nav = navigation.Navigator(MAP_W, MAP_H, stride=2)

running = False

def on_start():
    global running
    running = True
    print("▶ START")

def on_stop():
    global running
    running = False
    print("■ STOP")

keyboard.add_hotkey("F1", on_start)
keyboard.add_hotkey("F2", on_stop)
print("F1 시작 / F2 정지")

def wait_coord_change(before):
    """이동 후 좌표가 바뀌는지 일정 시간 대기. 바뀌면 새 좌표 반환, 아니면 None."""
    t0 = time.time()
    while time.time() - t0 < MOVE_CHECK_TIMEOUT:
        cur = vision.read_game_coord()
        if cur is not None and cur != before:
            return cur
        time.sleep(MOVE_CHECK_POLL)
    return None

while True:
    if not running:
        time.sleep(0.2)
        continue

    # 1) 게임 내부 좌표 읽기(필수)
    cur_cell = vision.read_game_coord()
    if cur_cell is None:
        # 좌표 못 읽으면 잠깐 대기 (인식 튜닝 필요)
        time.sleep(0.2)
        continue

    # 2) 화면상의 캐릭터 중심(우클릭 이동에 필요)
    char = vision.find_character_center()
    if not char:
        time.sleep(0.2)
        continue
    mx, my = char

    # 3) 몬스터 인식 및 전투 판단(기존)
    monsters = vision.find_monsters()
    near = 0
    far = 0
    target = None

    for x, y in monsters:
        d = max(abs(x-mx), abs(y-my))
        if d <= NEAR_RADIUS:
            near += 1
        else:
            far += 1
        if not target or d < max(abs(target[0]-mx), abs(target[1]-my)):
            target = (x, y)

    acts = combat.decision(state, near, far, target)

    # 4) 전투 액션 수행
    for a, t in acts:
        if a == "AIM":
            right_click(t[0], t[1])
        else:
            press_key(a)

    # 5) 전투 행동이 없으면 이동
    if not acts:
        d = nav.next_dir(cur_cell)
        if d is None:
            time.sleep(TICK)
            continue

        dx, dy = d

        # 이동 시도(화면 우클릭)
        move_dir(mx, my, dx, dy, MOVE_STEP)

        # 이동 성공/실패 판정: 좌표가 바뀌면 성공
        new_cell = wait_coord_change(cur_cell)
        if new_cell is None:
            # ❌ 좌표 변화 없음: 막힘 판정 -> 장애물 학습/우회
            nav.on_move_blocked(cur_cell, dx, dy)
        else:
            # ✅ 좌표 변화: 이동 성공
            nav.on_move_success(new_cell, MOVE_STEP)

    time.sleep(TICK)
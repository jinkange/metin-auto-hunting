
# main.py

from __future__ import annotations

import time
import threading
import keyboard

from config import (
    WINDOW_TITLE, MAP_W, MAP_H,
    MOVE_STEP_DEFAULT, MOVE_PIXEL_PER_TILE,
    TICK, NEAR_RADIUS,
    MOVE_CHECK_TIMEOUT, MOVE_CHECK_POLL
)

import windowing
import vision
import combat
import navigation
from events import DisappearWatcher
from input_ctrl import right_click, press_key


running = False
quit_flag = False
move_step = MOVE_STEP_DEFAULT


def on_start():
    global running
    running = True
    print("▶ START")


def on_stop():
    global running
    running = False
    print("■ STOP")


def wait_coord_change(grabber: vision.Grabber, before):
    t0 = time.time()
    while time.time() - t0 < MOVE_CHECK_TIMEOUT:
        cur = vision.read_game_coord(grabber)
        if cur is not None and cur != before:
            return cur
        time.sleep(MOVE_CHECK_POLL)
    return None


def console_loop(nav: navigation.Navigator):
    global quit_flag, move_step
    while not quit_flag:
        cmd = input(">> ").strip().lower()
        if cmd == "quit":
            quit_flag = True
        elif cmd.startswith("step "):
            try:
                n = int(cmd.split()[1])
                n = max(1, min(5, n))
                move_step = n
                print(f"[OK] move_step={move_step}")
            except:
                print("예: step 3")
        elif cmd.startswith("stride "):
            try:
                n = int(cmd.split()[1])
                n = max(1, min(10, n))
                nav.set_stride(n)
                print(f"[OK] stride={n}")
            except:
                print("예: stride 2")
        elif cmd == "status":
            print(f"running={running} move_step={move_step} path_i={nav.path_i} goal_i={nav.goal_i}")


def main():
    global quit_flag

    hwnd = windowing.find_window(WINDOW_TITLE)
    if not hwnd:
        print(f"[ERR] 창을 찾을 수 없습니다: {WINDOW_TITLE}")
        return

    windowing.bring_to_front(hwnd)
    windowing.move_window(hwnd, 0, 0)
    rect = windowing.get_window_rect(hwnd)

    grabber = vision.Grabber(rect)

    cstate = combat.CombatState()
    watcher = DisappearWatcher()
    nav = navigation.Navigator(MAP_W, MAP_H, stride=2)

    keyboard.add_hotkey("F1", on_start)
    keyboard.add_hotkey("F2", on_stop)

    threading.Thread(target=console_loop, args=(nav,), daemon=True).start()

    print("F1 시작 / F2 정지")
    print("콘솔: step N | stride N | status | quit")

    while not quit_flag:
        if not running:
            time.sleep(0.2)
            continue

        windowing.bring_to_front(hwnd)

        cur_cell = vision.read_game_coord(grabber)
        if cur_cell is None:
            time.sleep(0.1)
            continue

        win_gray = grabber.grab_gray()

        char = vision.find_character_center(win_gray)
        if char is None:
            time.sleep(0.1)
            continue
        cx, cy = char

        monsters = vision.find_monsters(win_gray)
        near = 0
        far = 0
        target = None

        for mx, my in monsters:
            dx_t = abs(mx - cx) / max(1, MOVE_PIXEL_PER_TILE)
            dy_t = abs(my - cy) / max(1, MOVE_PIXEL_PER_TILE)
            d = max(dx_t, dy_t)
            if d <= NEAR_RADIUS:
                near += 1
            else:
                far += 1
            if target is None or d < target[2]:
                target = (mx, my, d)

        target_xy = (target[0], target[1]) if target else None

        special_present = vision.check_special(win_gray)
        watcher.update(special_present)
        for key in watcher.pop_actions(max_n=3):
            press_key(key)

        acts = combat.decision(cstate, near, far, target_xy)
        for kind, val in acts:
            if kind == "AIM":
                right_click(val[0], val[1])
            elif kind == "PRESS":
                press_key(val)

        if not acts:
            dxy = nav.next_dir(cur_cell)
            if dxy is not None:
                dx, dy = dxy
                tx = cx + dx * MOVE_PIXEL_PER_TILE * move_step
                ty = cy + dy * MOVE_PIXEL_PER_TILE * move_step
                right_click(int(tx), int(ty))

                new_cell = wait_coord_change(grabber, cur_cell)
                if new_cell is None:
                    nav.on_move_blocked(cur_cell, dx, dy)
                else:
                    nav.on_move_success(new_cell, move_step)

        time.sleep(TICK)


if __name__ == "__main__":
    main()

# input_ctrl.py
import pyautogui
import time

pyautogui.FAILSAFE = False

def right_click(x, y):
    pyautogui.moveTo(x, y, duration=0.08)
    pyautogui.rightClick()

def press_key(k):
    pyautogui.press(str(k))

def move_dir(cx, cy, dx, dy, step):
    nx = cx + dx * step
    ny = cy + dy * step
    right_click(nx, ny)
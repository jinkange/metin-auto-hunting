
# input_ctrl.py
import pyautogui

pyautogui.FAILSAFE = False


def right_click(x: int, y: int):
    pyautogui.moveTo(x, y, duration=0.05)
    pyautogui.rightClick()


def press_key(k):
    pyautogui.press(str(k))

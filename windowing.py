
# windowing.py
import win32gui
import win32con


def find_window(title: str):
    hwnd = win32gui.FindWindow(None, title)
    if not hwnd:
        found = []

        def enum_cb(h, _):
            t = win32gui.GetWindowText(h)
            if title.lower() in t.lower() and win32gui.IsWindowVisible(h):
                found.append(h)

        win32gui.EnumWindows(enum_cb, None)
        hwnd = found[0] if found else None
    return hwnd


def bring_to_front(hwnd: int):
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32gui.SetForegroundWindow(hwnd)


def move_window(hwnd: int, x=0, y=0):
    l, t, r, b = win32gui.GetWindowRect(hwnd)
    w = r - l
    h = b - t
    win32gui.MoveWindow(hwnd, x, y, w, h, True)


def get_window_rect(hwnd: int):
    return win32gui.GetWindowRect(hwnd)

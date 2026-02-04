
# forbidden.py
from config import FORBIDDEN_POINTS, FORBIDDEN_PAD, FORBIDDEN_USE_CHEBYSHEV


def _dist(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) if FORBIDDEN_USE_CHEBYSHEV else (dx + dy)


def is_forbidden_cell(cell_xy):
    for fx, fy in FORBIDDEN_POINTS:
        if _dist(cell_xy, (fx, fy)) <= FORBIDDEN_PAD:
            return True
    return False

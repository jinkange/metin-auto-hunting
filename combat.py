
# combat.py
import random
from collections import deque


class CombatState:
    def __init__(self):
        self.last_569 = deque(maxlen=3)
        self.last_78 = None


def decision(state: CombatState, near_count: int, far_count: int, target_xy):
    # 규칙:
    # 1) 근처 몬스터 >= 3  -> 5,6,9 중 최근 사용과 중복되지 않게 1개
    # 2) 근처외 몬스터 >= 3 -> 7,8 번갈아
    # 3) 그 외 -> 타겟에 조준 후 4 한 번

    if near_count >= 3:
        cand = [5, 6, 9]
        opt = [c for c in cand if c not in state.last_569]
        k = random.choice(opt if opt else cand)
        state.last_569.append(k)
        return [("PRESS", k)]

    if far_count >= 3:
        if state.last_78 == 7:
            k = 8
        elif state.last_78 == 8:
            k = 7
        else:
            k = random.choice([7, 8])
        state.last_78 = k
        return [("PRESS", k)]

    if target_xy is not None:
        return [("AIM", target_xy), ("PRESS", 4)]

    return []

# combat.py
import random
from collections import deque

class CombatState:
    def __init__(self):
        self.last_569 = deque(maxlen=3)
        self.last_78 = None

def decision(state, near, far, target):
    if near >= 3:
        cand = [5,6,9]
        opt = [c for c in cand if c not in state.last_569]
        k = random.choice(opt if opt else cand)
        state.last_569.append(k)
        return [(k, None)]

    if far >= 3:
        k = 8 if state.last_78 == 7 else 7
        state.last_78 = k
        return [(k, None)]

    if target:
        return [("AIM", target), (4, None)]

    return []
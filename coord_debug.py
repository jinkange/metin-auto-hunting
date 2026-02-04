import time
import cv2
import numpy as np
import os
from pathlib import Path

import vision_coord_robust as v  # 아래에서 제공하는 파일
from config import COORD_REGION

OUT = Path("debug_coord")
OUT.mkdir(exist_ok=True)

print("[DEBUG] 3초 후 시작. 게임 창을 화면에 띄워두세요.")
time.sleep(3)

i = 0
while True:
    result = v.read_game_coord_debug(save_prefix=str(OUT / f"c{i:04d}"))
    print("coord =", result)
    i += 1
    time.sleep(0.2)
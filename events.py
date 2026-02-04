
# events.py
from queue import Queue


class DisappearWatcher:
    '''특수 이미지 3종 감시: True->False가 되면 이벤트 발생, 순서대로 1/2/3 입력'''

    def __init__(self):
        self.prev = [False, False, False]
        self.q = Queue()

    def update(self, present_list):
        for i in range(3):
            if self.prev[i] and (not present_list[i]):
                self.q.put(i + 1)
        self.prev = present_list[:]

    def pop_actions(self, max_n=3):
        out = []
        while (not self.q.empty()) and len(out) < max_n:
            out.append(self.q.get())
        return out

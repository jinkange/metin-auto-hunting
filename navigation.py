# navigation.py
import heapq
from forbidden import is_forbidden_cell

DIRS_8 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

def cheb(a,b):
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

class NavMap:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.occ = [[False]*w for _ in range(h)]
        self.vis = [[0]*w for _ in range(h)]

    def inb(self, x,y): return 0 <= x < self.w and 0 <= y < self.h
    def free(self, x,y):
        return self.inb(x,y) and (not self.occ[y][x]) and (not is_forbidden_cell((x,y)))

    def mark_occ(self, x,y):
        if self.inb(x,y): self.occ[y][x] = True

    def mark_vis(self, x,y):
        if self.inb(x,y): self.vis[y][x] += 1

def astar(nav: NavMap, start, goal):
    if start == goal: return [start]
    if not nav.free(*goal): return []

    def h(p): return float(cheb(p, goal))

    openh = []
    heapq.heappush(openh, (0.0, start))
    came = {start: None}
    g = {start: 0.0}

    while openh:
        _, cur = heapq.heappop(openh)
        if cur == goal:
            path = []
            p = cur
            while p is not None:
                path.append(p)
                p = came[p]
            return list(reversed(path))

        cx, cy = cur
        for dx, dy in DIRS_8:
            nx, ny = cx+dx, cy+dy
            if not nav.free(nx, ny):
                continue
            nxt = (nx, ny)
            cost = 1.4142 if (dx and dy) else 1.0
            ng = g[cur] + cost
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came[nxt] = cur
                heapq.heappush(openh, (ng + h(nxt), nxt))
    return []

def make_sweep_goals(w, h, stride=2):
    goals = []
    for y in range(0, h, stride):
        if (y//stride) % 2 == 0:
            xs = range(0, w, stride)
        else:
            xs = range(w-1, -1, -stride)
        for x in xs:
            goals.append((x,y))
    return goals

class Navigator:
    def __init__(self, w, h, stride=2):
        self.nav = NavMap(w,h)
        self.goals = make_sweep_goals(w,h,stride=stride)
        self.goal_i = 0
        self.path = []
        self.path_i = 0

    def next_goal(self):
        # 방문 적은 곳/가능한 곳을 찾기
        for _ in range(200):
            g = self.goals[self.goal_i % len(self.goals)]
            self.goal_i += 1
            if self.nav.free(*g):
                return g
        return self.goals[self.goal_i % len(self.goals)]

    def plan(self, cur):
        g = self.next_goal()
        self.path = astar(self.nav, cur, g)
        self.path_i = 0

    def next_dir(self, cur):
        # 경로 없으면 계획
        if not self.path or self.path_i >= len(self.path)-1:
            self.plan(cur)
        if not self.path or self.path_i >= len(self.path)-1:
            return None

        nx, ny = self.path[self.path_i + 1]
        dx = max(-1, min(1, nx - cur[0]))
        dy = max(-1, min(1, ny - cur[1]))
        return (dx, dy)

    def on_move_success(self, new_pos, step):
        self.nav.mark_vis(*new_pos)
        # path_i 대략 진행
        self.path_i = min(self.path_i + step, max(0, len(self.path)-1))

    def on_move_blocked(self, cur, dx, dy):
        # 막힌 방향 1칸을 장애물로 학습
        bx, by = cur[0] + dx, cur[1] + dy
        self.nav.mark_occ(bx, by)
        # 경로 재계획 유도
        self.path = []
        self.path_i = 0
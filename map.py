import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from typing import List, Tuple, Optional

from config import N_SQUARES, N_CIRCLES, ROBOT_RADIUS, R_PLAN, START, GOAL

Obstacle = tuple
Point    = Tuple[float, float]

X_MAX: int   = 40
Y_MAX: int   = 30
# Giá trị mặc định — đọc từ config.py (input_rrt_star.json)
_DEFAULT_START: Point = START
_DEFAULT_GOAL:  Point = GOAL

COLORS = {
    "wall":   "#7f8c8d",
    "shelf":  "#34495e",
    "pallet": "#d6d8d9",
    "square": "#e67e22",
    "circle": "#3498db",
}


# ══════════════════════════════════════════════════════════════════
#  PHẦN 1 – KIỂM TRA VA CHẠM
# ══════════════════════════════════════════════════════════════════

def point_in_rect(px: float, py: float, obs: Obstacle,
                  margin: float = 0.0) -> bool:
    # [FIX] Minkowski sum chính xác: disc bán kính margin với hình chữ nhật.
    # Tìm điểm gần nhất trên rect, kiểm tra khoảng cách ≤ margin.
    # Đúng hơn AABB vì xử lý góc bằng ¼ vòng tròn thay vì góc vuông.
    _, x, y, w, h = obs
    if margin == 0.0:
        return x <= px <= x + w and y <= py <= y + h
    closest_x = max(x, min(px, x + w))
    closest_y = max(y, min(py, y + h))
    return math.hypot(px - closest_x, py - closest_y) <= margin


def point_in_circle(px: float, py: float, obs: Obstacle,
                    margin: float = 0.0) -> bool:
    if obs[0] == 'circle':
        _, cx, cy, r = obs
        return math.hypot(px - cx, py - cy) <= r + margin
    return False


def point_in_obstacle(px: float, py: float, obs: Obstacle,
                      margin: float = 0.0) -> bool:
    if obs[0] == 'rect':
        return point_in_rect(px, py, obs, margin)
    return point_in_circle(px, py, obs, margin)


def point_in_any_obstacle(px: float, py: float,
                          obstacles: List[Obstacle],
                          margin: float = 0.0) -> bool:
    return any(point_in_obstacle(px, py, o, margin) for o in obstacles)


def segment_is_free(p1: Point, p2: Point,
                    obstacles: List[Obstacle],
                    robot_radius: float = 0.0,
                    n_samples: int = 40) -> bool:
    # [FIX] Số mẫu thích nghi: bước lấy mẫu ≤ min(robot_radius/2, 0.05m).
    # Điều này đảm bảo không bỏ sót vật cản nào có kích thước ≥ robot_radius/2.
    # Trước đây: n_samples cố định 40 → có thể bỏ sót vật cản nhỏ giữa 2 điểm mẫu.
    # Theo tinh thần Karaman & Frazzoli [4]: kiểm tra va chạm continuous. Sampling
    # thích nghi này là xấp xỉ tốt nhất mà không cần SOS trực tiếp trên edge.
    x1, y1 = p1
    x2, y2 = p2
    seg_len = math.hypot(x2 - x1, y2 - y1)
    if seg_len < 1e-9:
        return not point_in_any_obstacle(x1, y1, obstacles, robot_radius)

    if robot_radius > 0.0:
        step = min(robot_radius / 2.0, 0.05)
    else:
        step = 0.05
    n_adaptive = max(n_samples, math.ceil(seg_len / step))

    for i in range(n_adaptive + 1):
        t  = i / n_adaptive
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        if point_in_any_obstacle(px, py, obstacles, robot_radius):
            return False
    return True


def get_obstacle_bbox(obs: Obstacle) -> Tuple[float, float, float, float]:
    if obs[0] == 'rect':
        return obs[1], obs[2], obs[3], obs[4]
    if obs[0] == 'circle':
        _, cx, cy, r = obs
        return cx - r, cy - r, 2*r, 2*r
    raise ValueError(f"Unknown obstacle type: {obs[0]}")


def bbox_overlap(b1: Tuple, b2: Tuple) -> bool:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    return (x1 < x2 + w2 and x1 + w1 > x2 and
            y1 < y2 + h2 and y1 + h1 > y2)


# ══════════════════════════════════════════════════════════════════
#  PHẦN 2 – BẢN ĐỒ TĨNH
# ══════════════════════════════════════════════════════════════════

def build_static_obstacles() -> List[Obstacle]:
    obs: List[Obstacle] = []
    t = 0.5
    obs += [
        ('rect', 0,         0,         X_MAX, t    ),
        ('rect', 0,         Y_MAX - t, X_MAX, t    ),
        ('rect', 0,         0,         t,     Y_MAX),
        ('rect', X_MAX - t, 0,         t,     Y_MAX),
    ]
    obs += [
        ('rect', 0,    29.5, 12,  0.5),
        ('rect', 0,    20,   0.5, 9.5),
        ('rect', 11.5, 20,   0.5, 10 ),
        ('rect', 0,    20,   4,   0.5),
        ('rect', 7,    20,   5,   0.5),
    ]
    obs += [
        ('rect', 32,   19.5, 8,   0.5),
        ('rect', 39.5, 12,   0.5, 8  ),
        ('rect', 32,   12,   8,   0.5),
        ('rect', 32,   12,   0.5, 2  ),
        ('rect', 32,   17,   0.5, 3  ),
    ]
    for y in [2, 7, 12]:
        obs.append(('rect', 2,  y, 11, 2))
    for y in [2, 7, 12, 17, 22, 27]:
        obs.append(('rect', 16, y, 11, 2))
    obs.append(('rect', 29.5, 2, 1.5, 6))
    for px in [33.0, 35.0]:
        for py in [2.0, 4.5]:
            obs.append(('rect', px, py, 1.5, 1.5))
    for px in [29.5, 31.5]:
        for py in [23.0, 25.5, 28.0]:
            obs.append(('rect', px, py, 1.5, 1.5))
    return obs


# ══════════════════════════════════════════════════════════════════
#  PHẦN 3 – SINH VẬT CẢN NGẪU NHIÊN
# ══════════════════════════════════════════════════════════════════

def _placement_is_valid(new_bbox: Tuple,
                        existing: List[Obstacle],
                        start: Point,
                        goal: Point,
                        robot_radius: float = 0.4,   # [FIX 1] thêm robot_radius
                        sg_clearance: float = 1.5) -> bool:
    xn, yn, wn, hn = new_bbox

    # Quy tắc 1: cách Start và Goal tối thiểu sg_clearance
    for pt in (start, goal):
        if (xn - sg_clearance <= pt[0] <= xn + wn + sg_clearance and
                yn - sg_clearance <= pt[1] <= yn + hn + sg_clearance):
            return False

    # Quy tắc 2: [FIX 1] khoảng cách giữa vật cản đủ để robot đi qua
    min_gap = 2.0 * robot_radius + 0.2
    half    = min_gap / 2.0
    expanded = (xn - half, yn - half, wn + min_gap, hn + min_gap)
    for o in existing:
        if bbox_overlap(expanded, get_obstacle_bbox(o)):
            return False

    return True


def _gen_square(rng: random.Random,
                all_obs: List[Obstacle],
                pad: float,
                robot_radius: float,
                start: Point,
                goal: Point) -> Optional[Obstacle]:
    w = rng.uniform(1.0, 1.4)
    x = rng.uniform(pad, X_MAX - pad - w)
    y = rng.uniform(pad, Y_MAX - pad - w)
    bbox = (x, y, w, w)
    if _placement_is_valid(bbox, all_obs, start, goal, robot_radius):
        return ('rect', x, y, w, w)
    return None


def _gen_circle(rng: random.Random,
                all_obs: List[Obstacle],
                pad: float,
                robot_radius: float,
                start: Point,
                goal: Point) -> Optional[Obstacle]:
    r  = rng.uniform(0.4, 0.5)
    cx = rng.uniform(pad + r, X_MAX - pad - r)
    cy = rng.uniform(pad + r, Y_MAX - pad - r)
    bbox = (cx - r, cy - r, 2*r, 2*r)
    if _placement_is_valid(bbox, all_obs, start, goal, robot_radius):
        return ('circle', cx, cy, r)
    return None


def spawn_random_obstacles(
        existing:     List[Obstacle],
        n_squares:    int   = N_SQUARES,
        n_circles:    int   = N_CIRCLES,
        robot_radius: float = R_PLAN,
        seed:         Optional[int] = None,
        start:        Point = _DEFAULT_START,
        goal:         Point = _DEFAULT_GOAL,
) -> Tuple[List[Obstacle], int, int]:
    rng       = random.Random(seed)
    MAX_TRIES = 500
    PAD       = 0.5

    placed:  List[Obstacle] = []
    all_obs: List[Obstacle] = existing[:]

    def _try_place(generator_fn) -> bool:
        for _ in range(MAX_TRIES):
            result = generator_fn(rng, all_obs, PAD, robot_radius, start, goal)
            if result is not None:
                placed.append(result)
                all_obs.append(result)
                return True
        return False

    cnt_sq = 0
    cnt_cr = 0

    while cnt_sq < n_squares or cnt_cr < n_circles:
        if cnt_sq < cnt_cr and cnt_sq < n_squares:
            choice = 'square'
        elif cnt_cr < cnt_sq and cnt_cr < n_circles:
            choice = 'circle'
        else:
            choices = []
            if cnt_sq < n_squares: choices.append('square')
            if cnt_cr < n_circles: choices.append('circle')
            if not choices:
                break
            choice = rng.choice(choices)

        if choice == 'square':
            if _try_place(_gen_square):
                cnt_sq += 1
            else:
                print(f"  [MAP] Đặt được {cnt_sq}/{n_squares} hình vuông "
                      f"(hết chỗ, min_gap={2*robot_radius+0.2:.2f}m)")
                break
        else:
            if _try_place(_gen_circle):
                cnt_cr += 1
            else:
                print(f"  [MAP] Đặt được {cnt_cr}/{n_circles} hình tròn "
                      f"(hết chỗ, min_gap={2*robot_radius+0.2:.2f}m)")
                break

    return placed, cnt_sq, cnt_cr


# ══════════════════════════════════════════════════════════════════
#  PHẦN 4 – generate_map()
# ══════════════════════════════════════════════════════════════════

def generate_map(
        n_squares:    int           = N_SQUARES,
        n_circles:    int           = N_CIRCLES,
        robot_radius: float         = R_PLAN,
        seed:         Optional[int] = None,
        draw:         bool          = True,
        start:        Point         = _DEFAULT_START,
        goal:         Point         = _DEFAULT_GOAL,
) -> dict:
    static_obs = build_static_obstacles()
    dynamic_obs, cnt_sq, cnt_cr = spawn_random_obstacles(
        existing=static_obs, n_squares=n_squares,
        n_circles=n_circles, robot_radius=robot_radius,
        seed=seed, start=start, goal=goal,
    )
    all_obstacles = static_obs + dynamic_obs

    fig, ax, legend_handles = None, None, []
    if draw:
        fig, ax = _draw_map(all_obstacles, cnt_sq, cnt_cr, start, goal)
        legend_handles = _build_legend_handles(cnt_sq, cnt_cr)

    return {
        'obstacles':       all_obstacles,
        'start':           start,
        'goal':            goal,
        'x_max':           X_MAX,
        'y_max':           Y_MAX,
        'robot_radius':    robot_radius,
        'fig':             fig,
        'ax':              ax,
        'legend_handles':  legend_handles,
        'counts': {
            'squares': cnt_sq,
            'circles': cnt_cr,
            'total':   len(all_obstacles),
        },
    }


# ══════════════════════════════════════════════════════════════════
#  PHẦN 5 – VẼ BẢN ĐỒ
# ══════════════════════════════════════════════════════════════════

def _draw_map(obstacles, cnt_sq, cnt_cr,
              start: Point = _DEFAULT_START,
              goal:  Point = _DEFAULT_GOAL):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim([0, X_MAX]); ax.set_ylim([0, Y_MAX])
    ax.set_aspect('equal'); ax.set_facecolor('#f8f9fa')
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_title("Sơ đồ Nhà kho – RRT* Path Planning",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Trục X (m)"); ax.set_ylabel("Trục Y (m)")
    for obs in obstacles:
        _draw_obstacle(ax, obs)
    ax.plot(*start, marker='s', markersize=12, color='#2ECC71',
            markeredgecolor='black', zorder=10)
    ax.plot(*goal,  marker='*', markersize=16, color='#E74C3C',
            markeredgecolor='black', zorder=10)
    ax.annotate(f'Start {start}', xy=start,
                xytext=(start[0]+0.5, start[1]-1.5), fontsize=8)
    ax.annotate(f'Goal {goal}', xy=goal,
                xytext=(goal[0]+0.5, goal[1]+0.5), fontsize=8)
    return fig, ax


def _draw_obstacle(ax, obs):
    kind = obs[0]
    if kind == 'rect':
        _, x, y, w, h = obs
        color = COLORS["square"]
        if w >= X_MAX or h >= Y_MAX or abs(h - 0.5) < 1e-6 or abs(w - 0.5) < 1e-6:
            color = COLORS["wall"]
        elif abs(w - 1.5) < 1e-6 and abs(h - 1.5) < 1e-6:
            color = COLORS["pallet"]
        elif abs(w - 11) < 1e-6 or (abs(w - 1.5) < 1e-6 and h >= 6):
            color = COLORS["shelf"]
        ax.add_patch(mpatches.Rectangle((x, y), w, h,
                     facecolor=color, edgecolor='black',
                     linewidth=0.6, zorder=3))
    elif kind == 'circle':
        _, cx, cy, r = obs
        ax.add_patch(mpatches.Circle((cx, cy), r,
                     facecolor=COLORS["circle"], edgecolor='black',
                     linewidth=0.6, zorder=3))


def _build_legend_handles(cnt_sq, cnt_cr):
    return [
        mpatches.Patch(facecolor=COLORS["wall"],   edgecolor='black', label='Tường'),
        mpatches.Patch(facecolor=COLORS["shelf"],  edgecolor='black', label='Kệ hàng (Cố định)'),
        mpatches.Patch(facecolor=COLORS["pallet"], edgecolor='black', label='Bãi hàng (Cố định)'),
        mpatches.Patch(facecolor=COLORS["square"], edgecolor='black', label=f'Kiện hàng vuông ({cnt_sq})'),
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor=COLORS["circle"],
                      markeredgecolor='black', markersize=10, label=f'Vật tròn ({cnt_cr})'),
        mlines.Line2D([], [], marker='s', color='w', markerfacecolor='#2ECC71',
                      markeredgecolor='black', markersize=10, label='Start'),
        mlines.Line2D([], [], marker='*', color='w', markerfacecolor='#E74C3C',
                      markeredgecolor='black', markersize=15, label='Goal'),
    ]


def draw_rrt_tree(ax, nodes, parent, color='cyan', alpha=0.4, lw=0.5):
    for node in nodes:
        p = parent.get(node)
        if p is not None:
            ax.plot([node[0], p[0]], [node[1], p[1]],
                    color=color, alpha=alpha, linewidth=lw, zorder=2)


def draw_path(ax, path, reached_goal=True):
    if not path:
        return
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    if reached_goal:
        ax.plot(xs, ys, '-m', linewidth=3, zorder=8)
        ax.plot(xs, ys, 'mo', markersize=4, zorder=9, alpha=0.7)
    else:
        ax.plot(xs, ys, '--r', linewidth=2.5, zorder=8)


if __name__ == '__main__':
    _seed: int = 42
    env = generate_map(seed=_seed, robot_radius=0.55,
                       start=_DEFAULT_START, goal=_DEFAULT_GOAL)
    env['ax'].legend(handles=env['legend_handles'], loc='upper left',
                     bbox_to_anchor=(1, 1.02), fontsize=8)
    plt.tight_layout()
    print(f"Tổng obstacle: {env['counts']['total']}")
    plt.show()
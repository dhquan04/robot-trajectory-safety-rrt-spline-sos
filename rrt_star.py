import math
import random
from typing import List, Optional, Tuple, Dict

from config import (STEP_SIZE, MAX_ITER, GOAL_SAMPLE_RATE,
                    ROBOT_RADIUS, R_MAX, GAMMA_SCALE, GOAL_TOLERANCE)

# Obstacle type từ map.py
Obstacle = tuple
Point    = Tuple[float, float]


# ══════════════════════════════════════════════════════════════════
#  Node
# ══════════════════════════════════════════════════════════════════

class Node:
    __slots__ = ('x', 'y', 'parent', 'cost')

    def __init__(self, x: float, y: float):
        self.x:      float         = x
        self.y:      float         = y
        self.parent: Optional['Node'] = None
        self.cost:   float         = 0.0

    def as_point(self) -> Point:
        """Trả về (x, y) dưới dạng tuple thuần."""
        return (self.x, self.y)

    def __repr__(self):
        return f"Node({self.x:.2f}, {self.y:.2f}, cost={self.cost:.3f})"


# ══════════════════════════════════════════════════════════════════
#  Kiểm tra va chạm (tách riêng để dễ test và tái sử dụng với SOS)
# ══════════════════════════════════════════════════════════════════

def _point_collides(px: float, py: float,
                    obstacles: List[Obstacle],
                    robot_radius: float) -> bool:
    for obs in obstacles:
        t = obs[0]
        if t == 'rect':
            _, x, y, w, h = obs
            # [FIX] Minkowski sum chính xác: dist(điểm, hình chữ nhật) ≤ robot_radius.
            # Trước đây dùng AABB mở rộng (Minkowski với hình vuông), sai ở góc.
            closest_x = max(x, min(px, x + w))
            closest_y = max(y, min(py, y + h))
            if math.hypot(px - closest_x, py - closest_y) <= robot_radius:
                return True

        elif t == 'circle':
            _, cx, cy, r = obs
            if math.hypot(px - cx, py - cy) <= r + robot_radius:
                return True

    return False


def _segment_is_free(n1: Node, n2: Node,
                     obstacles: List[Obstacle],
                     robot_radius: float,
                     resolution: float = 0.05) -> bool:
    d = math.hypot(n2.x - n1.x, n2.y - n1.y)
    if d < 1e-9:
        return not _point_collides(n1.x, n1.y, obstacles, robot_radius)

    step = min(resolution, robot_radius / 2.0)
    n_samples = max(2, math.ceil(d / step))

    for i in range(n_samples + 1):
        t  = i / n_samples
        px = n1.x + t * (n2.x - n1.x)
        py = n1.y + t * (n2.y - n1.y)
        if _point_collides(px, py, obstacles, robot_radius):
            return False
    return True


# ══════════════════════════════════════════════════════════════════
#  Lớp RRTStar
# ══════════════════════════════════════════════════════════════════

class RRTStar:
    def __init__(
            self,
            start:            Point,
            goal:             Point,
            x_max:            float,
            y_max:            float,
            obstacles:        List[Obstacle],
            step_size:        float = STEP_SIZE,
            max_iter:         int   = MAX_ITER,
            goal_sample_rate: float = GOAL_SAMPLE_RATE,
            robot_radius:     float = ROBOT_RADIUS,
            r_max:            float = R_MAX,
            gamma_scale:      float = GAMMA_SCALE,
            goal_tolerance:   float = GOAL_TOLERANCE,
    ):
        # ── Cấu hình ──
        self.x_max            = x_max
        self.y_max            = y_max
        self.obstacles        = obstacles
        self.step_size        = step_size
        self.max_iter         = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.robot_radius     = robot_radius
        self.r_max            = r_max
        self.gamma_scale      = gamma_scale
        self.goal_tolerance   = goal_tolerance

        # ── Kiểm tra start/goal không bị va chạm ──
        if _point_collides(start[0], start[1], obstacles, robot_radius):
            raise ValueError(
                f"Start {start} nằm trong obstacle! "
                "Hãy kiểm tra lại vị trí Start."
            )
        if _point_collides(goal[0], goal[1], obstacles, robot_radius):
            raise ValueError(
                f"Goal {goal} nằm trong obstacle! "
                "Hãy kiểm tra lại vị trí Goal."
            )

        # ── Kiểm tra γ theo Karaman & Frazzoli (2011, IJRR Theorem 38) ───────
        # Điều kiện đủ để RRT* hội tụ tối ưu tiệm cận (asymptotically optimal):
        #
        #   γ_code  >  γ*  :=  (2·(1 + 1/d))^(1/d) · (μ_free / ζ_d)^(1/d)
        #
        # Với d = 2 chiều:
        #   ζ_2 = π  (diện tích hình tròn đơn vị bán kính 1)
        #   (2·(1+1/2))^(1/2) = (2·3/2)^(1/2) = √3
        #
        # ⟹ γ* = √3 · √(μ_free / π)
        #
        # Trong code: γ_code = gamma_scale · √(x_max · y_max / π)
        # Chặn trên an toàn: μ_free ≤ x_max · y_max
        # ⟹ γ*_ub = √3 · √(x_max · y_max / π)   [upper bound của γ*]
        #
        # Điều kiện: gamma_scale > √3 ≈ 1.732 là đủ trong trường hợp xấu nhất.
        # [4] Karaman, S. & Frazzoli, E. (2011). Sampling-based algorithms for
        #     optimal motion planning. IJRR 30(7), 846–894.
        _d          = 2
        _zeta_d     = math.pi            # diện tích hình tròn đơn vị (d=2)
        _mu_free_ub = x_max * y_max      # chặn trên diện tích tự do
        _gamma_star_ub = (
            (2.0 * (1.0 + 1.0 / _d)) ** (1.0 / _d)
            * (_mu_free_ub / _zeta_d) ** (1.0 / _d)
        )
        _gamma_code = gamma_scale * math.sqrt(x_max * y_max / math.pi)
        _ratio      = _gamma_code / _gamma_star_ub
        _status     = "✓ OK" if gamma_scale > math.sqrt(3.0) else "⚠️  DƯỚI MỨC TỐI THIỂU"
        print(
            f"[RRT*] γ_min (Karaman [4])  : {_gamma_star_ub:.4f}  "
            f"(gamma_scale tối thiểu ≈ √3 = {math.sqrt(3):.4f})\n"
            f"[RRT*] γ_used (scale={gamma_scale:.1f}): {_gamma_code:.4f}  "
            f"| tỉ lệ γ_used/γ_min = {_ratio:.2f}×  {_status}"
        )

        # ── Khởi tạo cây ──
        self.start = Node(start[0], start[1])
        self.goal  = Node(goal[0],  goal[1])

        self.node_list: List[Node]            = [self.start]
        self.parent:    Dict[Point, Optional[Point]] = {
            self.start.as_point(): None
        }
        # children dict: id(node) → list of child nodes
        # Duy trì liên tục để _propagate_cost không cần rebuild mỗi lần
        self._children: Dict[int, List[Node]] = {id(self.start): []}

    # ──────────────────────────────────────────────────────────────
    #  VÒNG LẶP CHÍNH
    # ──────────────────────────────────────────────────────────────

    def planning(self) -> Tuple[Optional[List[Point]], bool]:
        for i in range(self.max_iter):

            # Bước 1 – Sample
            q_rand = self._sample()

            # Bước 2 – Nearest
            q_nearest = self._nearest(q_rand)

            # Bước 3 – Steer (giới hạn step_size)
            q_new = self._steer(q_nearest, q_rand)

            # Bước 4 – Kiểm tra va chạm (liên tục)
            if not _segment_is_free(q_nearest, q_new,
                                    self.obstacles, self.robot_radius):
                continue   # đoạn này bị chặn → bỏ, lấy mẫu tiếp

            # Bước 5 – Tìm lân cận
            near_nodes = self._find_near_nodes(q_new)

            # Bước 6 – Chọn cha tốt nhất
            q_new = self._choose_parent(q_new, near_nodes)
            if q_new is None:
                continue   # không tìm được cha hợp lệ → bỏ qua

            # Bước 7 – Thêm vào cây
            self.node_list.append(q_new)
            self.parent[q_new.as_point()] = (
                q_new.parent.as_point() if q_new.parent else None
            )
            # Cập nhật children dict
            self._children[id(q_new)] = []
            if q_new.parent is not None:
                self._children.setdefault(id(q_new.parent), []).append(q_new)

            # Bước 8 – Rewire
            self._rewire(q_new, near_nodes)

        # Trích xuất đường đi tốt nhất
        return self._extract_best_path()

    # ──────────────────────────────────────────────────────────────
    #  SAMPLING
    # ──────────────────────────────────────────────────────────────

    def _sample(self) -> Node:
        if random.random() < self.goal_sample_rate:
            return Node(self.goal.x, self.goal.y)
        return Node(
            random.uniform(0, self.x_max),
            random.uniform(0, self.y_max),
        )

    # ──────────────────────────────────────────────────────────────
    #  NEAREST
    # ──────────────────────────────────────────────────────────────

    def _nearest(self, q_rand: Node) -> Node:
        best_dist_sq = float('inf')
        best_node    = self.node_list[0]
        for node in self.node_list:
            d_sq = (node.x - q_rand.x)**2 + (node.y - q_rand.y)**2
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_node    = node
        return best_node

    # ──────────────────────────────────────────────────────────────
    #  STEER
    # ──────────────────────────────────────────────────────────────

    def _steer(self, from_node: Node, to_node: Node) -> Node:
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d  = math.hypot(dx, dy)

        new_node = Node(from_node.x, from_node.y)

        if d <= self.step_size:
            # Đủ gần → di chuyển thẳng đến to_node
            new_node.x = to_node.x
            new_node.y = to_node.y
        else:
            # Giới hạn theo step_size
            theta      = math.atan2(dy, dx)
            new_node.x = from_node.x + self.step_size * math.cos(theta)
            new_node.y = from_node.y + self.step_size * math.sin(theta)

        # Tính cost thực tế = cost(from) + khoảng cách đã đi
        actual_dist   = math.hypot(new_node.x - from_node.x,
                                    new_node.y - from_node.y)
        new_node.cost   = from_node.cost + actual_dist
        new_node.parent = from_node
        return new_node

    # ──────────────────────────────────────────────────────────────
    #  FIND NEAR NODES
    # ──────────────────────────────────────────────────────────────

    def _find_near_nodes(self, q_new: Node) -> List[Node]:
        # Karaman & Frazzoli (2011) Theorem 38:
        #   r_n = γ · (log(n) / n)^(1/d),  n = số node hiện tại trong cây
        # q_new chưa được thêm vào node_list tại thời điểm gọi hàm này,
        # nên n = len(self.node_list) là đúng.
        # (Phiên bản cũ dùng n+1 → r_n hơi nhỏ hơn cần thiết, conservative)
        n = len(self.node_list)

        # γ chuẩn hoá theo diện tích không gian
        gamma = self.gamma_scale * math.sqrt(
            self.x_max * self.y_max / math.pi
        )
        r_n = min(gamma * math.sqrt(math.log(n) / n), self.r_max)

        r_n_sq = r_n ** 2
        return [
            node for node in self.node_list
            if (node.x - q_new.x)**2 + (node.y - q_new.y)**2 <= r_n_sq
        ]

    # ──────────────────────────────────────────────────────────────
    #  CHOOSE PARENT
    # ──────────────────────────────────────────────────────────────

    def _choose_parent(self, q_new: Node,
                       near_nodes: List[Node]) -> Optional[Node]:
        # ── Karaman & Frazzoli (2011) Algorithm 6 — ChooseParent ──────────────
        # q_nearest (= q_new.parent từ steer) đã được xác nhận collision-free ở
        # bước 4 của vòng lặp chính, nên LUÔN phải là ứng viên hợp lệ.
        # Khi n lớn, r_n → 0 và q_nearest có thể nằm ngoài near_nodes,
        # dẫn đến near_nodes không rỗng nhưng tất cả đều bị chặn → best_parent=None
        # → node bị discard dù đoạn q_nearest→q_new đã free.
        # Fix: gộp q_nearest vào near_nodes nếu chưa có.
        q_nearest = q_new.parent   # node từ _steer, đã qua collision check bước 4
        candidates = near_nodes if q_nearest in near_nodes else near_nodes + [q_nearest]

        best_cost   = float('inf')
        best_parent = None

        for candidate in candidates:
            d = math.hypot(q_new.x - candidate.x,
                           q_new.y - candidate.y)
            cost_via = candidate.cost + d

            if cost_via < best_cost:
                if _segment_is_free(candidate, q_new,
                                    self.obstacles, self.robot_radius):
                    best_cost   = cost_via
                    best_parent = candidate

        if best_parent is None:
            return None   # không tìm được cha hợp lệ (hiếm gặp)

        q_new.parent = best_parent
        q_new.cost   = best_cost
        return q_new

    # ──────────────────────────────────────────────────────────────
    #  REWIRE
    # ──────────────────────────────────────────────────────────────

    def _rewire(self, q_new: Node, near_nodes: List[Node]) -> None:
        for x in near_nodes:
            if x is q_new.parent:
                continue   # bỏ qua cha hiện tại (đã xử lý ở choose_parent)

            d = math.hypot(q_new.x - x.x, q_new.y - x.y)
            cost_via_new = q_new.cost + d

            if cost_via_new < x.cost:
                if _segment_is_free(q_new, x,
                                    self.obstacles, self.robot_radius):
                    # Xóa x khỏi children của cha cũ
                    old_parent = x.parent
                    if old_parent is not None:
                        old_children = self._children.get(id(old_parent), [])
                        if x in old_children:
                            old_children.remove(x)

                    # Gán cha mới
                    x.parent = q_new
                    x.cost   = cost_via_new

                    # Đồng bộ self.parent dict (dùng cho debug / export)
                    self.parent[x.as_point()] = q_new.as_point()

                    # Thêm x vào children của cha mới
                    self._children.setdefault(id(q_new), []).append(x)

                    self._propagate_cost(x)   # cập nhật cost toàn bộ cây con

    def _propagate_cost(self, parent: Node) -> None:
        from collections import deque

        queue = deque([parent])
        while queue:
            cur = queue.popleft()
            for child in self._children.get(id(cur), []):
                d          = math.hypot(child.x - cur.x, child.y - cur.y)
                child.cost = cur.cost + d
                queue.append(child)

    # ──────────────────────────────────────────────────────────────
    #  TRÍCH XUẤT ĐƯỜNG ĐI
    # ──────────────────────────────────────────────────────────────

    def _extract_best_path(
            self,
    ) -> Tuple[Optional[List[Point]], bool]:
        # ── Bước A: Tìm node trong vùng goal ──
        goal_tolerance_sq = self.goal_tolerance ** 2

        # FIX: dùng enumerate thay vì .index()
        candidates = [
            node for node in self.node_list
            if (node.x - self.goal.x)**2 + (node.y - self.goal.y)**2
               <= goal_tolerance_sq
        ]

        reached_goal = False
        end_node     = None

        if candidates:
            # Lọc thêm: đoạn cuối [candidate → goal] phải thông thoáng
            valid = [
                c for c in candidates
                if _segment_is_free(c, self.goal,
                                    self.obstacles, self.robot_radius)
            ]
            if valid:
                # Karaman & Frazzoli (2011): chọn theo TỔNG chi phí
                #   total_cost = cost_to_come(node) + cost_to_go(node → goal)
                # Không dùng min(cost) vì 2 node có cost giống nhau nhưng
                # khoảng cách đến goal khác nhau → path dài khác nhau.
                end_node = min(
                    valid,
                    key=lambda n: n.cost + math.hypot(
                        n.x - self.goal.x, n.y - self.goal.y
                    )
                )
                reached_goal = True

        if end_node is None:
            # Bước B: không tới đích → lấy node gần nhất
            end_node = min(
                self.node_list,
                key=lambda n: (n.x - self.goal.x)**2 + (n.y - self.goal.y)**2
            )
            print(f"[RRT*] Chưa tới đích. Node gần nhất: "
                  f"({end_node.x:.2f}, {end_node.y:.2f}), "
                  f"dist = {math.hypot(end_node.x - self.goal.x, end_node.y - self.goal.y):.3f}")

        # ── Truy ngược cây để lấy path ──
        path: List[Point] = []

        if reached_goal:
            # Cập nhật mới: Chỉ thêm điểm đích nếu end_node chưa nằm trúng đích
            if end_node.as_point() != self.goal.as_point():
                path.append(self.goal.as_point())

        node = end_node
        while node is not None:
            path.append(node.as_point())
            node = node.parent

        path.reverse()   # đảo: [start, ..., goal]

        if len(path) < 2:
            return None, False

        return path, reached_goal
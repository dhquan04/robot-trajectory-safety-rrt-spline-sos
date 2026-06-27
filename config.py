"""
config.py — Nguồn thông số DUY NHẤT của toàn bộ hệ thống.

Đọc input_rrt_star.json một lần, export các hằng số module-level.
Mọi file khác (rrt_star.py, map.py, main.py, ...) import từ đây
thay vì hardcode giá trị — chỉ cần sửa JSON là đủ.
"""

import json
import os

_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "input_rrt_star.json")


def _load() -> dict:
    if not os.path.exists(_JSON_PATH):
        raise FileNotFoundError(
            f"[config] Không tìm thấy '{_JSON_PATH}'.\n"
            "Hãy đặt input_rrt_star.json cùng thư mục với config.py."
        )
    with open(_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_cfg = _load()

# ════════════════════════════════════════════════════════
#  ROBOT
# ════════════════════════════════════════════════════════
ROBOT_RADIUS:  float = _cfg["robot"]["radius"]
SAFETY_MARGIN: float = _cfg["robot"].get("safety_margin", 0.0)
R_PLAN:        float = ROBOT_RADIUS + SAFETY_MARGIN        # bán kính kế hoạch
START:         tuple = tuple(_cfg["robot"]["start"])
GOAL:          tuple = tuple(_cfg["robot"]["goal"])

# ════════════════════════════════════════════════════════
#  RRT*
# ════════════════════════════════════════════════════════
STEP_SIZE:        float = _cfg["rrt_param"]["step_size"]
MAX_ITER:         int   = _cfg["rrt_param"]["max_iter"]
GOAL_SAMPLE_RATE: float = _cfg["rrt_param"]["goal_sample_rate"]
GOAL_TOLERANCE:   float = _cfg["rrt_param"]["goal_tolerance"]
GAMMA_SCALE:      float = _cfg["rrt_param"].get("gamma_scale", 2.0)
R_MAX:            float = _cfg["rrt_param"].get("r_max", 5.0)

# ════════════════════════════════════════════════════════
#  MAP
# ════════════════════════════════════════════════════════
N_SQUARES: int             = _cfg["map"]["n_squares"]
N_CIRCLES: int             = _cfg["map"]["n_circles"]
SEED:      int | None      = _cfg["map"]["seed"]
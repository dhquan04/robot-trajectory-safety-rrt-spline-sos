import sys, csv, json, math, os, time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import map as env_map
from rrt_star import RRTStar, Node, _segment_is_free

INPUT_FILE = "input_rrt_star.json"

# ══════════════════════════════════════════════════════════════════
#  ĐỌC THAM SỐ ĐẦU VÀO
# ══════════════════════════════════════════════════════════════════

def load_config(filepath=INPUT_FILE):
    if not os.path.exists(filepath):
        print(f"  [LỖI] Không tìm thấy file: '{filepath}'")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"  [CONFIG] Đã đọc tham số từ '{filepath}'")
    return cfg

# ══════════════════════════════════════════════════════════════════
#  HÀM PHỤ TRỢ
# ══════════════════════════════════════════════════════════════════

def compute_path_length(path):
    if path is None or len(path) < 2:
        return 0.0
    return sum(math.hypot(path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
               for i in range(1, len(path)))

def print_summary(cfg, env, path, reached_goal, n_nodes, elapsed):
    sep = "─" * 55
    print(f"\n{'═'*55}")
    print(f"  KẾT QUẢ RRT* PATH PLANNING")
    print(f"{'═'*55}")
    print(f"  Bản đồ       : {env['x_max']} × {env['y_max']} m")
    print(f"  Seed         : {cfg['map']['seed']}")
    print(f"  Start        : {cfg['robot']['start']}")
    print(f"  Goal         : {cfg['robot']['goal']}")
    print(f"  Robot radius : {cfg['robot']['radius']} m")
    print(sep)
    print(f"  Số obstacle  : {env['counts']['total']}"
          f"  (■ {env['counts']['squares']} ● {env['counts']['circles']})")
    print(f"  Số node RRT* : {n_nodes}")
    print(f"  Thời gian    : {elapsed:.4f} s")          # hiển thị 4 chữ số
    print(sep)
    if path is not None:
        status = "✅ Tới đích!" if reached_goal else "⚠️  Chưa tới đích"
        print(f"  Kết quả      : {status}")
        print(f"  Số waypoint  : {len(path)}")
        print(f"  Độ dài path  : {compute_path_length(path):.4f} m")  # hiển thị 4 chữ số
    else:
        print("  Kết quả      : ❌ Không tìm được đường đi")
    print(f"{'═'*55}\n")

# ══════════════════════════════════════════════════════════════════
#  LƯU KẾT QUẢ RA CSV  (full precision – KHÔNG làm tròn)
# ══════════════════════════════════════════════════════════════════

def save_outputs(cfg, env, path_raw, reached_goal, n_nodes, elapsed,
                 map_attempt=1, used_seed=None,
                 elapsed_per_attempt=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = os.path.join("outputs", f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Tính arc-length KHÔNG làm tròn
    def arc_lengths(path):
        if not path:
            return []
        s = [0.0]
        for i in range(1, len(path)):
            ds = math.hypot(path[i][0] - path[i-1][0],
                            path[i][1] - path[i-1][1])
            s.append(s[-1] + ds)          # ← full float64, không round
        return s

    # ── 1. waypoints.csv ─────────────────────────────────────────
    wp_file = os.path.join(out_dir, "waypoints.csv")
    with open(wp_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "x", "y", "arc_length"])
        if path_raw:
            s_values = arc_lengths(path_raw)
            for p, s_val in zip(path_raw, s_values):
                # repr(float) giữ đủ chữ số để khôi phục chính xác
                writer.writerow(["raw", repr(p[0]), repr(p[1]), repr(s_val)])

    # ── 2. obstacles.csv ─────────────────────────────────────────
    def obs_label(obs):
        if obs[0] == 'circle':
            return 'circle'
        # rect tuple: ('rect', x, y, w, h) hoặc ('rect', x, y, w, h, label)
        if len(obs) >= 6 and isinstance(obs[5], str):
            return obs[5]   # ← dùng label map.py đã gán
        # Fallback: suy ra từ kích thước — đúng logic map.py
        _, x, y, w, h = obs[:5]
        X_MAX, Y_MAX = env_map.X_MAX, env_map.Y_MAX
        if (w >= X_MAX or h >= Y_MAX
                or abs(w - 0.5) < 1e-6
                or abs(h - 0.5) < 1e-6):    return 'wall'
        if abs(w - 1.5) < 1e-6 and abs(h - 1.5) < 1e-6:
                                             return 'pallet'
        if abs(w - 11) < 1e-6 or (abs(w - 1.5) < 1e-6 and h >= 6):
                                             return 'shelf'
        return 'square'

    with open(os.path.join(out_dir, "obstacles.csv"),
              "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "type", "p1", "p2", "p3", "p4", "label"])
        for i, obs in enumerate(env["obstacles"]):
            if obs[0] == 'rect':
                label = obs_label(obs)
                _, ox, oy, ow, oh = obs[:5]
                w.writerow([i, "rect", ox, oy, ow, oh, label])
            elif obs[0] == 'circle':
                _, cx, cy, r = obs
                w.writerow([i, "circle", cx, cy, r, "", "circle"])

    # ── 3. metadata.csv ──────────────────────────────────────────
    path_length = compute_path_length(path_raw)
    rows = [
        ["run",       "timestamp",        datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["run",       "input_file",        INPUT_FILE],
        ["run",       "elapsed_rrt_s",     elapsed],
        ["run",       "map_attempts",       map_attempt],
        ["run",       "seed_used",          used_seed],
        ["robot",     "start_x",           cfg["robot"]["start"][0]],
        ["robot",     "start_y",           cfg["robot"]["start"][1]],
        ["robot",     "goal_x",            cfg["robot"]["goal"][0]],
        ["robot",     "goal_y",            cfg["robot"]["goal"][1]],
        ["robot",     "radius",            cfg["robot"]["radius"]],
        ["robot",     "safety_margin",     cfg["robot"].get("safety_margin", 0.0)],
        ["robot",     "r_plan",            cfg["robot"]["radius"] + cfg["robot"].get("safety_margin", 0.0)],
        ["rrt_param", "max_iter",          cfg["rrt_param"]["max_iter"]],
        ["rrt_param", "step_size",         cfg["rrt_param"]["step_size"]],
        ["rrt_param", "goal_sample_rate",  cfg["rrt_param"]["goal_sample_rate"]],
        ["rrt_param", "goal_tolerance",    cfg["rrt_param"]["goal_tolerance"]],
        ["rrt_param", "gamma_scale",       cfg["rrt_param"].get("gamma_scale", 2.0)],
        ["rrt_param", "r_max",             cfg["rrt_param"].get("r_max", 5.0)],
        ["map",       "x_max",             env["x_max"]],
        ["map",       "y_max",             env["y_max"]],
        ["map",       "seed",              str(cfg["map"]["seed"])],
        ["map",       "n_squares",         env["counts"]["squares"]],
        ["map",       "n_circles",         env["counts"]["circles"]],
        ["map",       "n_obstacles_total", env["counts"]["total"]],
        ["result",    "reached_goal",      reached_goal],
        ["result",    "n_nodes_rrt",       n_nodes],
        ["path_raw",  "n_waypoints",       len(path_raw) if path_raw else 0],
        ["path_raw",  "length_m",          path_length],           # full precision
    ]
    # ── Thời gian từng lần thử sinh bản đồ ───────────────────────
    if elapsed_per_attempt:
        for i, t in enumerate(elapsed_per_attempt, start=1):
            rows.append(["run", f"elapsed_map{i}_s", t])
    with open(os.path.join(out_dir, "metadata.csv"),
              "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "key", "value"])
        w.writerows(rows)

    abs_dir = os.path.abspath(out_dir)
    print(f"\n  {'─'*53}")
    print(f"  [OUTPUT] Đã lưu 3 file CSV →  {abs_dir}/")
    print(f"  {'─'*53}")
    return out_dir




def main():
    print("=" * 55)
    print("  KHỞI ĐỘNG RRT* PATH PLANNING")
    print("=" * 55)

    # ── Bước 1: Đọc JSON ─────────────────────────────────────────
    cfg          = load_config()
    robot        = cfg["robot"]
    rrt_param    = cfg["rrt_param"]
    map_param    = cfg["map"]
    start        = tuple(robot["start"])
    goal         = tuple(robot["goal"])
    robot_radius = robot["radius"]
    safety_margin = robot.get("safety_margin", 0.0)
    r_plan = robot_radius + safety_margin

    # ── Bước 2+3: Tạo bản đồ → Chạy RRT* → Sinh lại nếu thất bại ──
    #
    #  Chiến lược: nếu RRT* không tìm được đường với bản đồ hiện tại,
    #  KHÔNG tăng max_iter (tốn thời gian vô ích trên bản đồ khó),
    #  mà sinh bản đồ MỚI với seed ngẫu nhiên khác và thử lại từ đầu.
    #  Mỗi lần sinh bản đồ mới chỉ mất < 1 giây.
    #
    #  Tham số:
    #    MAX_MAP_RETRIES : số lần tối đa sinh lại bản đồ
    #    FIXED_SEED      : seed cố định nếu JSON dùng null (None = random)
    # ─────────────────────────────────────────────────────────────────
    MAX_MAP_RETRIES = 10   # tối đa 10 bản đồ khác nhau
    import random as _rnd

    print("  [1-2/4] Tạo bản đồ và chạy RRT* …")
    print(f"          max_iter={rrt_param['max_iter']:,} | "
          f"step={rrt_param['step_size']} | "
          f"r_plan={r_plan}m | "
          f"γ_scale={rrt_param.get('gamma_scale', 2.0)} | "
          f"r_max={rrt_param.get('r_max', 5.0)}m | "
          f"tối đa {MAX_MAP_RETRIES} bản đồ")

    path                = None
    reached_goal        = False
    planner             = None
    elapsed             = 0.0
    elapsed_per_attempt = []           # ← thời gian từng lần thử sinh bản đồ
    env                 = None
    used_seed           = map_param["seed"]   # None hoặc int từ JSON
    map_attempt         = 0

    # Nếu seed=null trong JSON → tự sinh seed ngẫu nhiên lần đầu
    if used_seed is None:
        used_seed = _rnd.randint(0, 999_999)

    while map_attempt < MAX_MAP_RETRIES:
        map_attempt += 1
        print(f"\n  ── Bản đồ {map_attempt}/{MAX_MAP_RETRIES}"
              f" | seed = {used_seed} ──")

        # Tạo bản đồ với seed hiện tại
        plt.close("all")   # đóng figure cũ tránh tràn bộ nhớ
        env = env_map.generate_map(
            n_squares    = map_param["n_squares"],
            n_circles    = map_param["n_circles"],
            robot_radius = r_plan,
            seed         = used_seed,
            draw         = False,   # chỉ vẽ khi tìm được đường
            start        = start,
            goal         = goal,
        )
        print(f"       {env['counts']['total']} obstacles "
              f"(■ {env['counts']['squares']} vuông,"
              f" ● {env['counts']['circles']} tròn)")

        # Chạy RRT* trên bản đồ này
        try:
            planner = RRTStar(
                start            = start,
                goal             = goal,
                x_max            = env["x_max"],
                y_max            = env["y_max"],
                obstacles        = env["obstacles"],
                step_size        = rrt_param["step_size"],
                max_iter         = rrt_param["max_iter"],
                goal_sample_rate = rrt_param["goal_sample_rate"],
                robot_radius     = r_plan,
                goal_tolerance   = rrt_param["goal_tolerance"],
                gamma_scale      = rrt_param.get("gamma_scale", 2.0),
                r_max            = rrt_param.get("r_max", 5.0),
            )
        except ValueError as e:
            print(f"       [LỖI planner] {e} → thử bản đồ khác")
            used_seed = _rnd.randint(0, 999_999)
            continue

        t0 = time.time()
        path, reached_goal = planner.planning()
        t_attempt = time.time() - t0
        elapsed  += t_attempt
        elapsed_per_attempt.append(t_attempt)   # ← lưu thời gian lần này
        n_nodes = len(planner.node_list)

        print(f"       {elapsed:.1f}s | {n_nodes:,} nodes", end="")

        if reached_goal:
            print(f" | ✅ Tìm được đường ({len(path)} waypoints)")
            break
        else:
            print(f" | ❌ Bản đồ khó → sinh bản đồ mới …")
            used_seed = _rnd.randint(0, 999_999)   # seed mới hoàn toàn

    if not reached_goal:
        print(f"\n  ⛔ Thất bại sau {MAX_MAP_RETRIES} bản đồ "
              f"(tổng {elapsed:.1f}s).")

    # Vẽ bản đồ thành công (hoặc bản đồ cuối cùng nếu thất bại)
    plt.close("all")
    env = env_map.generate_map(
        n_squares    = map_param["n_squares"],
        n_circles    = map_param["n_circles"],
        robot_radius = r_plan,
        seed         = used_seed,
        draw         = True,
        start        = start,
        goal         = goal,
    )
    ax = env["ax"]

    path_raw = list(path) if path else None

    # Ghi lại seed đã dùng để tái tạo bản đồ nếu cần
    cfg["map"]["seed_used"] = used_seed

    # ── Bước 4: Lưu CSV ──────────────────────────────────────────
    out_dir = save_outputs(cfg, env, path_raw, reached_goal,
                           len(planner.node_list) if planner else 0,
                           elapsed, map_attempt, used_seed,
                           elapsed_per_attempt)

    # ── Bước 5: Vẽ ───────────────────────────────────────────────
    print("  [3/4] Đang vẽ kết quả …")
    env_map.draw_path(ax, path_raw, reached_goal)

    legend_handles = env["legend_handles"][:]
    if reached_goal:
        legend_handles.append(
            mlines.Line2D([], [], color='magenta', linewidth=2.5,
                          label=f'Đường đi RRT* ( {len(path_raw)} wp, '
                                f'bản đồ #{map_attempt}, seed={used_seed})'))
    else:
        legend_handles.append(
            mlines.Line2D([], [], color='red', linewidth=2.0, linestyle='--',
                          label=f'Đường đi RRT* (chưa tới đích, {len(path_raw) if path_raw else 0} wp)'))
    ax.legend(handles=legend_handles, loc='upper left',
              bbox_to_anchor=(1.0, 1.02), fontsize=8, framealpha=0.9)
    plt.tight_layout()

    fig = plt.gcf()
    img_path = os.path.join(out_dir, "path_planning_result.png")
    fig.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"  [OUTPUT] Đã lưu HÌNH ẢNH →  {img_path}")

    # ── Bước 6: Thống kê ─────────────────────────────────────────
    print("  [4/4] Hoàn tất.")
    print_summary(cfg, env, path_raw, reached_goal,
                  len(planner.node_list) if planner else 0, elapsed)

    if path_raw:
        print("  Path (danh sách waypoint [x, y]):")
        for i, (px, py) in enumerate(path_raw):
            print(f"    [{i:03d}]  x={px:.4f},  y={py:.4f}")   # hiển thị 4 chữ số

    plt.show()

if __name__ == '__main__':
    main()
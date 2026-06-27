import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ══════════════════════════════════════════════════════════════════
#  TÌM THƯ MỤC RUN MỚI NHẤT
# ══════════════════════════════════════════════════════════════════

def get_latest_run_dir():
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục '{output_dir}'")
    subdirs = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("run_")
    ]
    if not subdirs:
        raise FileNotFoundError("Không có thư mục run_ nào trong outputs/")
    return max(subdirs, key=os.path.getmtime)


# ══════════════════════════════════════════════════════════════════
#  THUẬT TOÁN THOMAS  (full float64 – KHÔNG làm tròn trung gian)
# ══════════════════════════════════════════════════════════════════

def solve_thomas(a, b, c, d):
    n   = len(d)
    c_p = np.zeros(n - 1, dtype=np.float64)
    d_p = np.zeros(n,     dtype=np.float64)

    # Khử xuôi
    c_p[0] = c[0] / b[0]
    d_p[0] = d[0] / b[0]
    for k in range(1, n):
        denom = b[k] - a[k - 1] * c_p[k - 1]
        if k < n - 1:
            c_p[k] = c[k] / denom
        d_p[k] = (d[k] - a[k - 1] * d_p[k - 1]) / denom

    # Thế ngược
    x = np.zeros(n, dtype=np.float64)
    x[-1] = d_p[-1]
    for k in range(n - 2, -1, -1):
        x[k] = d_p[k] - c_p[k] * x[k + 1]

    return x


# ══════════════════════════════════════════════════════════════════
#  TÍNH MOMENT  (Natural Spline: M₀ = Mₙ = 0)
# ══════════════════════════════════════════════════════════════════

def compute_moments(s, P):
    h = np.diff(s)                               # float64

    diag_main  = 2.0 * (h[:-1] + h[1:])
    diag_sub   = h[1:-1].copy()
    diag_super = h[1:-1].copy()
    rhs = 6.0 * ((P[2:] - P[1:-1]) / h[1:]
                 - (P[1:-1] - P[:-2]) / h[:-1])

    M_internal = solve_thomas(diag_sub, diag_main, diag_super, rhs)
    return np.concatenate(([0.0], M_internal, [0.0]))


# ══════════════════════════════════════════════════════════════════
#  TÍNH HỆ SỐ POWER FORM
# ══════════════════════════════════════════════════════════════════

def compute_coefficients(s, P, M):
    h = np.diff(s)
    a = P[:-1].copy()
    c = M[:-1] / 2.0
    d = (M[1:] - M[:-1]) / (6.0 * h)
    b = (P[1:] - P[:-1]) / h - h * (2.0 * M[:-1] + M[1:]) / 6.0
    return a, b, c, d


# ══════════════════════════════════════════════════════════════════
#  ĐÁNH GIÁ SPLINE TẠI s BẤT KỲ
# ══════════════════════════════════════════════════════════════════

def evaluate_spline(s_nodes, a, b, c, d, s_query):
    idx = int(np.clip(
        np.searchsorted(s_nodes, s_query, side='right') - 1,
        0, len(a) - 1
    ))
    t = s_query - s_nodes[idx]
    return a[idx] + b[idx]*t + c[idx]*t**2 + d[idx]*t**3


# ══════════════════════════════════════════════════════════════════
#  VẼ VẬT CẢN  (đồng bộ màu sắc với map.py)
# ══════════════════════════════════════════════════════════════════

_COLORS = {
    'wall':   ('#7f8c8d', 0.90),
    'shelf':  ('#34495e', 0.85),
    'pallet': ('#d6d8d9', 0.90),
    'square': ('#e67e22', 0.80),
    'circle': ('#3498db', 0.70),
}

_LABEL_DISPLAY = {
    'wall':   'Tường',
    'shelf':  'Kệ hàng',
    'pallet': 'Bãi hàng',
    'square': 'Kiện hàng vuông',
    'circle': 'Vật tròn',
}

_X_MAX, _Y_MAX = 40, 30   # fallback — bị ghi đè khi đọc metadata.csv


def _load_map_bounds(run_dir):
    """Đọc x_max, y_max từ metadata.csv; trả về fallback (40, 30) nếu thiếu."""
    meta_path = os.path.join(run_dir, "metadata.csv")
    if not os.path.exists(meta_path):
        return 40, 30
    x_max = y_max = None
    with open(meta_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('key') == 'x_max':
                x_max = float(row['value'])
            elif row.get('key') == 'y_max':
                y_max = float(row['value'])
    return (x_max or 40), (y_max or 30)


def _infer_label(obs_type, w, h, x_max=_X_MAX, y_max=_Y_MAX):
    """Suy nhãn từ kích thước — đúng logic map.py."""
    if obs_type == 'circle':
        return 'circle'
    if w >= x_max or h >= y_max or abs(h - 0.5) < 1e-6 or abs(w - 0.5) < 1e-6:
        return 'wall'
    if abs(w - 1.5) < 1e-6 and abs(h - 1.5) < 1e-6:
        return 'pallet'
    if abs(w - 11) < 1e-6 or (abs(w - 1.5) < 1e-6 and h >= 6):
        return 'shelf'
    return 'square'


def _load_obstacles(run_dir):
    """Đọc obstacles.csv, trả về list dict tương thích map.py."""
    path = os.path.join(run_dir, "obstacles.csv")
    if not os.path.exists(path):
        return []
    x_max, y_max = _load_map_bounds(run_dir)
    obs = []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            t = row['type']
            if t == 'circle':
                w = h = float(row['p3']) * 2
            else:
                w, h = float(row['p3']), float(row['p4'])
            lbl = row.get('label', '').strip() or _infer_label(t, w, h, x_max, y_max)
            o = {'type': t, 'label': lbl}
            if t == 'circle':
                o.update({'cx': float(row['p1']), 'cy': float(row['p2']),
                          'r':  float(row['p3'])})
            else:
                o.update({'x_bl': float(row['p1']), 'y_bl': float(row['p2']),
                          'w': float(row['p3']), 'h_r': float(row['p4'])})
            obs.append(o)
    return obs


def _draw_obstacles(ax, obstacles):
    """Vẽ vật cản lên ax, trả về legend patches."""
    legend_patches = {}
    for obs in obstacles:
        lbl = obs['label']
        color, alpha = _COLORS.get(lbl, ('#888888', 0.75))
        if obs['type'] == 'circle':
            patch = mpatches.Circle(
                (obs['cx'], obs['cy']), obs['r'],
                facecolor=color, edgecolor='black',
                linewidth=0.6, alpha=alpha, zorder=2)
        else:
            patch = mpatches.Rectangle(
                (obs['x_bl'], obs['y_bl']), obs['w'], obs['h_r'],
                facecolor=color, edgecolor='black',
                linewidth=0.6, alpha=alpha, zorder=2)
        ax.add_patch(patch)
        if lbl not in legend_patches:
            legend_patches[lbl] = mpatches.Patch(
                facecolor=color, edgecolor='black', linewidth=0.6,
                alpha=alpha, label=_LABEL_DISPLAY.get(lbl, lbl.capitalize()))
    return list(legend_patches.values())


def verify_c2_continuity(s, ax_c, bx_c, cx_c, dx_c,
                          ay_c, by_c, cy_c, dy_c):
    """
    Xác minh số học tính liên tục C² tại mỗi nút nội (junction node).

    Với Natural Cubic Spline, các điều kiện C⁰, C¹, C² tại nút s_{i+1}:

        C⁰: x(s_{i+1}⁻) = ax_{i+1}           (hiển nhiên theo định nghĩa)
        C¹: x'(s_{i+1}⁻) = bx_{i+1}
              = bx_i + 2·cx_i·h_i + 3·dx_i·h_i²
        C²: x''(s_{i+1}⁻) = 2·cx_{i+1}
              = 2·cx_i + 6·dx_i·h_i

    Residual lý thuyết = 0 (chính xác về mặt đại số).
    Residual số học ~ ε_machine (≈ 1e-15) — nếu lớn hơn 1e-10 cần kiểm tra lại.
    Tương tự cho y(s).

    Trả về: list of dict, mỗi dict chứa chỉ số junction và 6 residuals.
    """
    n_seg    = len(s) - 1
    results  = []

    for i in range(n_seg - 1):
        h_i = float(s[i + 1] - s[i])

        # --- Giới hạn bên phải tại s[i+1]: τ = h_i (cuối đoạn i) ---
        x_r0 = ax_c[i] + bx_c[i]*h_i + cx_c[i]*h_i**2 + dx_c[i]*h_i**3
        x_r1 = bx_c[i] + 2.0*cx_c[i]*h_i + 3.0*dx_c[i]*h_i**2
        x_r2 = 2.0*cx_c[i] + 6.0*dx_c[i]*h_i

        y_r0 = ay_c[i] + by_c[i]*h_i + cy_c[i]*h_i**2 + dy_c[i]*h_i**3
        y_r1 = by_c[i] + 2.0*cy_c[i]*h_i + 3.0*dy_c[i]*h_i**2
        y_r2 = 2.0*cy_c[i] + 6.0*dy_c[i]*h_i

        # --- Giới hạn bên trái tại s[i+1]: τ = 0 (đầu đoạn i+1) ---
        x_l0 = ax_c[i + 1]
        x_l1 = bx_c[i + 1]
        x_l2 = 2.0 * cx_c[i + 1]

        y_l0 = ay_c[i + 1]
        y_l1 = by_c[i + 1]
        y_l2 = 2.0 * cy_c[i + 1]

        results.append({
            'junction': i + 1,
            's':        float(s[i + 1]),
            'err_C0_x': abs(x_r0 - x_l0),
            'err_C1_x': abs(x_r1 - x_l1),
            'err_C2_x': abs(x_r2 - x_l2),
            'err_C0_y': abs(y_r0 - y_l0),
            'err_C1_y': abs(y_r1 - y_l1),
            'err_C2_y': abs(y_r2 - y_l2),
        })

    return results


def print_c2_table(c2_rows):
    """In bảng residuals C² ra terminal."""
    header = (f"  {'Jct':>3}  {'s':>8}  "
              f"{'|ΔC0_x|':>10}  {'|ΔC1_x|':>10}  {'|ΔC2_x|':>10}  "
              f"{'|ΔC0_y|':>10}  {'|ΔC1_y|':>10}  {'|ΔC2_y|':>10}")
    sep = "  " + "─" * (len(header) - 2)
    print("\n  [C²-VERIFY] Kiểm tra tính liên tục tại các nút nội:")
    print(sep)
    print(header)
    print(sep)
    max_c2 = 0.0
    for r in c2_rows:
        e = [r['err_C0_x'], r['err_C1_x'], r['err_C2_x'],
             r['err_C0_y'], r['err_C1_y'], r['err_C2_y']]
        max_c2 = max(max_c2, r['err_C2_x'], r['err_C2_y'])
        print(f"  {r['junction']:>3}  {r['s']:>8.4f}  "
              + "  ".join(f"{v:>10.2e}" for v in e))
    print(sep)
    status = "✓ OK (≤ 1e-10)" if max_c2 <= 1e-10 else f"⚠️  max C² err = {max_c2:.2e}"
    print(f"  max |ΔC²| = {max_c2:.2e}   {status}\n")



def main():
    print("=" * 55)
    print("  NATURAL CUBIC SPLINE  –  Thomas Algorithm")
    print("=" * 55)

    # ── 1. Đọc waypoints (full precision) ────────────────────────
    run_dir = get_latest_run_dir()
    wp_path = os.path.join(run_dir, "waypoints.csv")
    print(f"  [1/4] Đọc waypoints:\n        {wp_path}")

    s_list, x_list, y_list = [], [], []
    with open(wp_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s_list.append(float(row["arc_length"]))   # float() giữ full precision
            x_list.append(float(row["x"]))
            y_list.append(float(row["y"]))

    s = np.array(s_list, dtype=np.float64)
    x = np.array(x_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    n_seg = len(s) - 1
    print(f"       {len(s)} nút, {n_seg} đoạn")

    # ── 2. Tính Moment (full float64) ────────────────────────────
    print("  [2/4] Giải Moment M (Thomas) …")
    Mx = compute_moments(s, x)
    My = compute_moments(s, y)
    # In 4 chữ số để đọc, KHÔNG dùng giá trị này trong tính toán
    print(f"       Mx = {[round(v, 4) for v in Mx.tolist()]}")
    print(f"       My = {[round(v, 4) for v in My.tolist()]}")

    # ── 3. Tính hệ số (full float64) ─────────────────────────────
    print("  [3/4] Tính hệ số a,b,c,d …")
    ax_c, bx_c, cx_c, dx_c = compute_coefficients(s, x, Mx)
    ay_c, by_c, cy_c, dy_c = compute_coefficients(s, y, My)

    # ── Xác minh C² continuity tại mỗi nút nội ──────────────────
    c2_rows = verify_c2_continuity(s, ax_c, bx_c, cx_c, dx_c,
                                    ay_c, by_c, cy_c, dy_c)
    print_c2_table(c2_rows)

    # In bảng hệ số: 4 chữ số để đọc
    print(f"\n  {'─'*53}")
    print(f"  {'seg':>3}  {'s_start':>9}  {'s_end':>9}  "
          f"{'ax':>9}  {'bx':>9}  {'cx':>9}  {'dx':>9}")
    print(f"  {'─'*53}")
    for i in range(n_seg):
        print(f"  {i:>3}  {s[i]:>9.4f}  {s[i+1]:>9.4f}  "
              f"{ax_c[i]:>9.4f}  {bx_c[i]:>9.4f}  "
              f"{cx_c[i]:>9.4f}  {dx_c[i]:>9.4f}")
    print(f"  {'─'*53}\n")

    # ── 4. Lưu CSV (full precision – repr) ───────────────────────
    out_file = os.path.join(run_dir, "spline_equations.csv")
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "segment", "s_start", "s_end", "h",
            "ax", "bx", "cx", "dx",
            "ay", "by", "cy", "dy",
            "Mx_start", "Mx_end", "My_start", "My_end"
        ])
        for i in range(n_seg):
            writer.writerow([
                i,
                repr(float(s[i])),       repr(float(s[i+1])),
                repr(float(s[i+1] - s[i])),
                repr(float(ax_c[i])),    repr(float(bx_c[i])),
                repr(float(cx_c[i])),    repr(float(dx_c[i])),
                repr(float(ay_c[i])),    repr(float(by_c[i])),
                repr(float(cy_c[i])),    repr(float(dy_c[i])),
                repr(float(Mx[i])),      repr(float(Mx[i+1])),
                repr(float(My[i])),      repr(float(My[i+1])),
            ])
    print(f"  [4/4] Lưu spline_equations.csv → {out_file}")

    # ── 5. Vẽ kiểm tra ───────────────────────────────────────────
    s_fine = np.linspace(s[0], s[-1], 2000)
    x_fine = np.array([evaluate_spline(s, ax_c, bx_c, cx_c, dx_c, si)
                       for si in s_fine])
    y_fine = np.array([evaluate_spline(s, ay_c, by_c, cy_c, dy_c, si)
                       for si in s_fine])

    # ── Đọc obstacles ─────────────────────────────────────────────
    obstacles = _load_obstacles(run_dir)
    x_max_map, y_max_map = _load_map_bounds(run_dir)
    has_obs   = len(obstacles) > 0
    if has_obs:
        print(f"  [viz] Đọc {len(obstacles)} vật cản từ obstacles.csv")
    else:
        print("  [viz] Không tìm thấy obstacles.csv — vẽ không có vật cản")

    fig, ax0 = plt.subplots(figsize=(10, 9))

    # ── Quỹ đạo 2D (có vật cản) ──────────────────────────────────
    ax0.set_facecolor('#f8f9fa')

    # Vẽ vật cản trước (zorder thấp)
    obs_patches = []
    if has_obs:
        obs_patches = _draw_obstacles(ax0, obstacles)

    # Vẽ spline và waypoints lên trên
    ax0.plot(x_fine, y_fine, color='#1E88E5', linewidth=2.2,
             label='Cubic Spline', zorder=5)
    ax0.plot(x, y, 'o--', color='#E53935', markersize=4,
             linewidth=0.8, label='Waypoints RRT*', zorder=6)
    ax0.plot(x[0],  y[0],  's', color='#2ECC71', markersize=10,
             markeredgecolor='black', label='Start', zorder=7)
    ax0.plot(x[-1], y[-1], '*', color='#E74C3C', markersize=13,
             markeredgecolor='black', label='Goal',  zorder=7)

    ax0.set_xlabel("X (m)"); ax0.set_ylabel("Y (m)")
    ax0.set_title("Natural Cubic Spline – Kiểm tra kết quả",
                  fontsize=13, fontweight='bold', pad=10)
    ax0.set_xlim(0, x_max_map); ax0.set_ylim(0, y_max_map)
    ax0.set_aspect('equal')
    ax0.grid(True, linestyle='--', alpha=0.4)

    # Legend: đưa ra ngoài plot, neo vào cạnh phải ax
    spline_handles, spline_labels = ax0.get_legend_handles_labels()
    all_handles = spline_handles + obs_patches
    all_labels  = spline_labels  + [p.get_label() for p in obs_patches]
    ax0.legend(all_handles, all_labels, fontsize=8,
               loc='upper left',
               bbox_to_anchor=(1.01, 1.0),
               borderaxespad=0,
               framealpha=0.9)

    plt.tight_layout()
    img_path = os.path.join(run_dir, "spline_result.png")
    fig.savefig(img_path, dpi=200, bbox_inches='tight')

    print(f"\n  {'─'*53}")
    print(f"  Số đoạn        : {n_seg}")
    print(f"  Arc length tổng: {s[-1]:.4f} m")       # hiển thị 4 chữ số
    print(f"  CSV            : spline_equations.csv")
    print(f"  Hình ảnh       : spline_result.png")
    print(f"  {'─'*53}\n")

    plt.show()


if __name__ == "__main__":
    main()
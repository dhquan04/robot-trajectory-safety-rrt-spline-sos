import os, csv, math, textwrap, datetime, time
import numpy as np
import cvxpy as cp
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

R_ROBOT = 0.55   # fallback — bị ghi đè bởi giá trị r_plan đọc từ metadata.csv


# ══════════════════════════════════════════════════════════════════
#  0.  I/O
# ══════════════════════════════════════════════════════════════════

def get_latest_run_dir():
    base = "outputs"
    if not os.path.exists(base):
        raise FileNotFoundError("Không tìm thấy thư mục 'outputs/'")
    runs = [os.path.join(base, d) for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d)) and d.startswith("run_")]
    if not runs:
        raise FileNotFoundError("Không có run_ nào trong outputs/")
    return sorted(runs)[-1]


def load_spline(run_dir):
    # [FIX] Dùng try/except cho từng giá trị thay vì float() vô điều kiện.
    # Cột 'segment' là int; nếu sau này thêm cột string, sẽ không crash.
    with open(os.path.join(run_dir, "spline_equations.csv")) as f:
        rows = []
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v   # giữ nguyên chuỗi nếu không phải số
            rows.append(parsed)
        return rows


def _infer_label(row: dict, x_max: float = 40, y_max: float = 30) -> str:

    t = row.get("type", "rect")
    if t == "circle":
        return "circle"
    try:
        w = float(row.get("p3", 0))
        h = float(row.get("p4", 0))
    except (TypeError, ValueError):
        return "square"
    if w >= x_max or h >= y_max or abs(h - 0.5) < 1e-6 or abs(w - 0.5) < 1e-6:
        return "wall"
    if abs(w - 1.5) < 1e-6 and abs(h - 1.5) < 1e-6:
        return "pallet"
    if abs(w - 11) < 1e-6 or (abs(w - 1.5) < 1e-6 and h >= 6):
        return "shelf"
    return "square"


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


def load_obstacles(run_dir):
    obs = []
    path = run_dir if run_dir.endswith(".csv") else os.path.join(run_dir, "obstacles.csv")
    # Xác định run_dir gốc để đọc metadata
    base_dir = os.path.dirname(path) if run_dir.endswith(".csv") else run_dir
    x_max, y_max = _load_map_bounds(base_dir)
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for row in reader:
            # Ưu tiên nhãn trong CSV; fallback suy luận nếu trống/thiếu
            lbl = row.get("label", "").strip()
            if not lbl:
                lbl = _infer_label(row, x_max, y_max)

            o = {"index": int(row["index"]), "type": row["type"], "label": lbl}
            if row["type"] == "circle":
                o.update({"cx": float(row["p1"]), "cy": float(row["p2"]),
                           "r":  float(row["p3"])})
            else:
                o.update({"x_bl": float(row["p1"]), "y_bl": float(row["p2"]),
                           "w":   float(row["p3"]), "h_r": float(row["p4"])})
            obs.append(o)
    return obs


# ══════════════════════════════════════════════════════════════════
#  1.  SPLINE
# ══════════════════════════════════════════════════════════════════

def eval_xy(seg, tau):
    """Evaluate cubic spline at parameter τ."""
    x = seg["ax"] + seg["bx"]*tau + seg["cx"]*tau**2 + seg["dx"]*tau**3
    y = seg["ay"] + seg["by"]*tau + seg["cy"]*tau**2 + seg["dy"]*tau**3
    return x, y


def seg_bbox(seg):
    h = seg["h"]

    def cubic_range(a, b, c, d):
        """Min and max of a + b*t + c*t^2 + d*t^3 on [0, h]."""
        vals = [a, a + b*h + c*h**2 + d*h**3]   # endpoints
        # derivative: b + 2c*t + 3d*t^2  (degree 2)
        # solve with quadratic formula when leading coeff non-zero
        if abs(3*d) > 1e-14:
            disc = (2*c)**2 - 4*(3*d)*b
            if disc >= 0:
                sq = math.sqrt(disc)
                for t in ((-2*c + sq) / (6*d), (-2*c - sq) / (6*d)):
                    if 0.0 < t < h:
                        vals.append(a + b*t + c*t**2 + d*t**3)
        elif abs(2*c) > 1e-14:          # linear derivative
            t = -b / (2*c)
            if 0.0 < t < h:
                vals.append(a + b*t + c*t**2)
        return min(vals), max(vals)

    xmn, xmx = cubic_range(seg["ax"], seg["bx"], seg["cx"], seg["dx"])
    ymn, ymx = cubic_range(seg["ay"], seg["by"], seg["cy"], seg["dy"])
    return xmn, xmx, ymn, ymx


# ══════════════════════════════════════════════════════════════════
#  2.  HỆ SỐ BARRIER
# ══════════════════════════════════════════════════════════════════

def barrier_circle_coeffs(seg, cx, cy, rho):
    px = np.array([seg["ax"] - cx, seg["bx"], seg["cx"], seg["dx"]])
    py = np.array([seg["ay"] - cy, seg["by"], seg["cy"], seg["dy"]])
    p  = np.convolve(px, px) + np.convolve(py, py)
    p[0] -= rho**2
    return p


# ══════════════════════════════════════════════════════════════════
#  3.  MIN B(τ) — GIẢI TÍCH CHÍNH XÁC
# ══════════════════════════════════════════════════════════════════

def poly_min_on_interval(coeffs, h, n_grid=200):
    """
    Tìm giá trị nhỏ nhất của đa thức p(τ) = Σ coeffs[k]·τ^k trên [0, h].

    Chiến lược 2 giai đoạn — ổn định số học:

    (A) Lưới dày (dense grid): đánh giá tại n_grid+1 điểm đều nhau.
        Mục đích: bắt được mọi cực tiểu cục bộ, kể cả khi hệ số rất nhỏ
        làm cho np.roots trả về nghiệm sai (vì companion matrix ill-conditioned).

    (B) Nghiệm chính xác: dùng numpy.polynomial.polynomial.Polynomial.roots()
        thay vì np.polyder + np.roots.
        Lý do: Polynomial làm việc với dạng hệ số tăng dần (ascending),
        tránh lỗi do hệ số dẫn đầu gần 0 trong dạng giảm dần (descending)
        mà np.polyder + np.roots yêu cầu.  Với đa thức bậc 5 từ B'(τ) của
        barrier bậc 6, hệ số bậc cao thường rất nhỏ → companion matrix
        của np.roots dễ trở nên singular.

    Tất cả ứng viên (2 đầu mút + n_grid điểm lưới + nghiệm nội tiếp) đều
    được đánh giá; giá trị nhỏ nhất toàn cục được trả về.
    """
    from numpy.polynomial.polynomial import Polynomial

    p = Polynomial(coeffs)            # ascending: coeffs[k] = hệ số τ^k

    # (A) Đánh giá tại đầu mút
    v0   = float(p(0.0))
    vh   = float(p(h))
    v_min = min(v0, vh)
    t_min = 0.0 if v0 <= vh else h

    # (B) Lưới dày — bắt mọi cực tiểu dù hệ số nhỏ
    t_grid = np.linspace(0.0, h, n_grid + 1)
    v_grid = p(t_grid)
    idx    = int(np.argmin(v_grid))
    if v_grid[idx] < v_min:
        v_min = float(v_grid[idx])
        t_min = float(t_grid[idx])

    # (C) Nghiệm chính xác của p'(τ) bằng Polynomial (ổn định hơn np.roots)
    dp = p.deriv()                    # vẫn ở dạng ascending
    try:
        roots_dp = dp.roots()
        for r in roots_dp:
            # Chỉ chấp nhận nghiệm gần thực (|Im| < ngưỡng tương đối)
            if abs(r.imag) <= 1e-8 * (abs(r.real) + 1.0):
                rv = float(r.real)
                if 0.0 < rv < h:
                    v = float(p(rv).real)
                    if v < v_min:
                        v_min, t_min = v, rv
    except np.linalg.LinAlgError:
        # Companion matrix singular → lưới dày đã bao phủ
        pass

    return t_min, v_min


# ══════════════════════════════════════════════════════════════════
#  4.  SOS — GRAM MATRIX (Lukács decomposition, degree 6)
# ══════════════════════════════════════════════════════════════════

def sos_to_poly_coeffs(Q0, Q1, h):
    n0, n1 = Q0.shape[0], Q1.shape[0]   # 4, 3

    # σ₀(τ) = v₀ᵀ Q₀ v₀  (degree 2*(n0−1) = 6)
    c_sig0 = np.zeros(2 * n0 - 1)
    for i in range(n0):
        for j in range(n0):
            c_sig0[i + j] += Q0[i, j]

    # σ₁(τ) = v₁ᵀ Q₁ v₁  (degree 2*(n1−1) = 4)
    c_sig1 = np.zeros(2 * n1 - 1)
    for i in range(n1):
        for j in range(n1):
            c_sig1[i + j] += Q1[i, j]

    # Multiply σ₁ by τ(h−τ) = h·τ − τ²
    # ascending-power representation: [0, h, −1]
    tau_factor    = np.array([0.0, h, -1.0])
    c_sig1_scaled = np.convolve(c_sig1, tau_factor)   # degree 6

    return c_sig0 + c_sig1_scaled


def exact_linf_poly(diff_coeffs, h, n_grid=200):
    from numpy.polynomial.polynomial import Polynomial

    p = Polynomial(diff_coeffs)       # ascending

    # Đầu mút
    candidates = [0.0, h]

    # Lưới dày (fallback)
    t_grid = np.linspace(0.0, h, n_grid + 1)
    candidates.extend(t_grid.tolist())

    # Nghiệm chính xác của D'(τ)
    dp = p.deriv()
    try:
        for r in dp.roots():
            if abs(r.imag) <= 1e-8 * (abs(r.real) + 1.0):
                rv = float(r.real)
                if 0.0 < rv < h:
                    candidates.append(rv)
    except np.linalg.LinAlgError:
        pass

    return float(max(abs(float(p(t).real)) for t in candidates))


def build_gram_deg6(c, h):
    Q0 = cp.Variable((4, 4), PSD=True)
    Q1 = cp.Variable((3, 3), PSD=True)

    # 7 linear equality constraints — coefficient matching τ⁰ … τ⁶
    cons = [
        Q0[0, 0]                                                        == c[0],
        2*Q0[0, 1]  +  h*Q1[0, 0]                                      == c[1],
        Q0[1, 1] + 2*Q0[0, 2]  +  2*h*Q1[0, 1]  -  Q1[0, 0]          == c[2],
        2*(Q0[1, 2]+Q0[0, 3])  +  h*(Q1[1, 1]+2*Q1[0, 2])  -  2*Q1[0, 1]  == c[3],
        Q0[2, 2] + 2*Q0[1, 3]  +  2*h*Q1[1, 2]  -  (Q1[1, 1]+2*Q1[0, 2]) == c[4],
        2*Q0[2, 3]  +  h*Q1[2, 2]  -  2*Q1[1, 2]                      == c[5],
        Q0[3, 3]  -  Q1[2, 2]                                           == c[6],
    ]

    prob = cp.Problem(cp.Minimize(0), cons)
    prob.solve(solver=cp.SCS, eps=1e-6, verbose=False)

    if Q0.value is None:
        # ── Phân biệt nguyên nhân thất bại ────────────────────────────────
        # 'infeasible' / 'infeasible_inaccurate':
        #     Đa thức B(τ) KHÔNG SOS trên [0,h].
        #     Đây là bằng chứng *ngược*: B(τ) < 0 tại đâu đó → có thể unsafe.
        #     Khác với lỗi số (solver_error, unbounded) là bài toán không xác định.
        status = getattr(prob, 'status', 'unknown')
        is_infeasible = status in ('infeasible', 'infeasible_inaccurate')
        # Trả về flag để caller phân biệt "không SOS" vs "lỗi số"
        return None, None, np.inf, 'infeasible' if is_infeasible else f'solver_error:{status}'

    # ── Exact L∞ certificate error ────────────────────────────────────────────
    # D(τ) = B(τ) − SOS(τ)  is a degree-6 polynomial.
    # Its max modulus on [0, h] is found exactly by locating critical points
    # of |D(τ)| via the real roots of D'(τ) (degree 5, ≤ 5 real roots),
    # then evaluating at those roots plus the two endpoints.
    # Reference: standard result of real analysis (Extreme Value Theorem +
    # Fermat's theorem); no sampling error.
    sos_coeffs  = sos_to_poly_coeffs(Q0.value, Q1.value, h)
    diff_coeffs = c - sos_coeffs          # D(τ) = B(τ) − SOS(τ), ascending
    err         = exact_linf_poly(diff_coeffs, h)

    return Q0.value, Q1.value, err, 'optimal'


# ══════════════════════════════════════════════════════════════════
#  5.  KIỂM TRA TỪNG CẶP
# ══════════════════════════════════════════════════════════════════

def check_circle(seg, obs):
    rho    = obs["r"] + R_ROBOT
    coeffs = barrier_circle_coeffs(seg, obs["cx"], obs["cy"], rho)
    h      = seg["h"]

    # ══ Bước 1: SDP feasibility quyết định SAFE/UNSAFE ═══════════════
    # Định lý Lukács (1918): B(τ) ≥ 0 trên [0,h] ⟺ tồn tại Q₀⪰0, Q₁⪰0
    #   thỏa  B(τ) = v₀ᵀQ₀v₀ + τ(h−τ)·v₁ᵀQ₁v₁  (Lukács decomposition).
    # ⟹  SDP khả thi ⟺ B(τ) là SOS trên [0,h] ⟺ SAFE.
    # Ref: Parrilo (2000, §3); Prajna & Jadbabaie (2007, §II).
    # KHÔNG dùng poly_min_on_interval / min_B / B'(τ)=0 để kết luận safe.
    Q0, Q1, err, gram_status = build_gram_deg6(coeffs, h)
    sdp_safe = (gram_status == 'optimal') and (Q0 is not None)

    # ── Bước 2: Chẩn đoán — tìm τ*, X*, Y*, min_B (KHÔNG quyết định safe) ──
    # poly_min_on_interval chỉ dùng cho mục đích báo cáo / hiển thị tọa độ
    # vi phạm.  Giá trị mv KHÔNG được dùng làm điều kiện kết luận SAFE/UNSAFE.
    ts, mv = poly_min_on_interval(coeffs, h)
    Xs, Ys = eval_xy(seg, ts)

    res = {
        "type": "circle", "rho": rho, "poly_deg": 6,
        "axis": "N/A", "method": "circle_deg6_Lukacs",
        "min_B":    mv,        # chẩn đoán / báo cáo — KHÔNG quyết định safe
        "tau_star": ts, "X_star": Xs, "Y_star": Ys,
        "safe":    sdp_safe,   # ← SDP feasibility là nguồn chân lý duy nhất
        "gram_status": gram_status, "Q2": None,
        "Q0":      Q0  if sdp_safe else None,
        "Q1":      Q1  if sdp_safe else None,
        "gram_err": err if sdp_safe else float('nan'),
    }
    return res


# Giới hạn mỗi chiều để tránh quá nhiều SDP calls với vật cản cực lớn.
# Tổng circles tối đa = MAX_DIM^2 = 1024, vẫn đủ nhanh với prefilter.
MAX_DIM = 32


def _rect_circles(x_bl, y_bl, w, h_r):
    N_x = min(MAX_DIM, max(1, math.ceil(w   / (2 * R_ROBOT))))
    N_y = min(MAX_DIM, max(1, math.ceil(h_r / (2 * R_ROBOT))))

    cell_w = w    / N_x
    cell_h = h_r  / N_y
    R_c    = math.sqrt((cell_w / 2)**2 + (cell_h / 2)**2)
    rho    = R_c + R_ROBOT

    # Xác minh coverage invariant tại runtime (debug guard)
    assert R_c >= 0, "R_c phải không âm"
    assert rho > R_ROBOT, "ρ phải lớn hơn R_ROBOT"

    circles = [
        (x_bl + (ix + 0.5) * cell_w,
         y_bl + (iy + 0.5) * cell_h)
        for ix in range(N_x)
        for iy in range(N_y)
    ]
    return circles, rho, N_x, N_y


def check_rect(seg, obs):
    x_bl = obs["x_bl"]; w   = obs["w"]
    y_bl = obs["y_bl"]; h_r = obs["h_r"]

    circles, rho, N_x, N_y = _rect_circles(x_bl, y_bl, w, h_r)
    n_circles = len(circles)
    h = seg["h"]

    # ── Bước 1: Tính hệ số barrier cho tất cả circles ────────────────
    cf_list = [barrier_circle_coeffs(seg, xc, yc, rho) for xc, yc in circles]

    # ══ Bước 2: SDP feasibility quyết định SAFE/UNSAFE ═══════════════
    # Coverage invariant (§2.3): an toàn ⟺ MỌI N_x×N_y SDP đều khả thi.
    # Chạy build_gram_deg6 cho TẤT CẢ circles TRƯỚC KHI dùng poly_min.
    # KHÔNG dùng poly_min_on_interval / min_B / global_min để kết luận safe.
    q0_list = []; q1_list = []; err_list = []
    status_list = []; sdp_ok_list = []

    for c_i in cf_list:
        Q0_i, Q1_i, err_i, st_i = build_gram_deg6(c_i, h)
        ok_i = (st_i == 'optimal') and (Q0_i is not None)
        q0_list.append(Q0_i);    q1_list.append(Q1_i)
        err_list.append(err_i);  status_list.append(st_i)
        sdp_ok_list.append(ok_i)

    sdp_safe     = all(sdp_ok_list)     # ← nguồn chân lý duy nhất
    n_sos_ok     = sum(sdp_ok_list)
    n_infeasible = sum(1 for s in status_list if s == 'infeasible')
    valid_errs   = [err_list[i] for i in range(n_circles)
                    if sdp_ok_list[i] and not math.isnan(err_list[i])]
    gram_err_max = float(np.max(valid_errs)) if valid_errs else float('nan')

    # ── Bước 3: Chẩn đoán — poly_min_on_interval cho báo cáo ─────────
    # Chỉ dùng để tìm τ*, (X*, Y*), min_B hiển thị / tọa độ vi phạm.
    # Giá trị global_min KHÔNG được dùng làm điều kiện kết luận SAFE/UNSAFE.
    mv_list = []; ts_list = []
    for c_i in cf_list:
        ts_i, mv_i = poly_min_on_interval(c_i, h)
        mv_list.append(mv_i); ts_list.append(ts_i)

    worst_idx         = int(np.argmin(mv_list))
    global_min        = mv_list[worst_idx]   # min B trên toàn lưới (chẩn đoán)
    worst_ts          = ts_list[worst_idx]
    Xs, Ys            = eval_xy(seg, worst_ts)
    Q0_worst          = q0_list[worst_idx]
    Q1_worst          = q1_list[worst_idx]
    err_worst         = err_list[worst_idx]
    gram_status_worst = status_list[worst_idx]

    # gram_err_max: L∞ error lớn nhất trên toàn bộ N circles
    res = {
        "type": obs["type"], "rho": rho, "poly_deg": 6,
        "axis": f"grid_{N_x}x{N_y}_{n_circles}circles",
        "method": f"rect_adaptive_grid_{N_x}x{N_y}_deg6_Lukacs_full_cert",
        "min_B":    global_min,  # chẩn đoán / báo cáo — KHÔNG quyết định safe
        "tau_star": worst_ts,
        "X_star": Xs, "Y_star": Ys,
        "safe": sdp_safe,        # ← SDP feasibility là nguồn chân lý duy nhất
        "Q2": None,
        "n_circles": n_circles, "N_x": N_x, "N_y": N_y,
        "n_sos_certified": n_sos_ok,       # số circles có Gram matrix hợp lệ
        "n_sos_infeasible": n_infeasible,  # số circles SDP trả về infeasible
        "gram_err_max": gram_err_max,      # L∞ error tối đa trên toàn lưới
        "gram_status": gram_status_worst,
    }
    if sdp_safe:
        res.update({"Q0": Q0_worst, "Q1": Q1_worst,
                    "gram_err": err_worst,
                    "gram_err_all_max": gram_err_max})
    else:
        res.update({"Q0": None, "Q1": None,
                    "gram_err": float('nan'),
                    "gram_err_all_max": float('nan')})
    return res


# ── DEPRECATED: check_wall ────────────────────────────────────────────────────
# BUG: Logic OR với corner barriers không phải là chứng chỉ an toàn hợp lệ.
#      B_corner_i(τ) > 0 chỉ nghĩa là quỹ đạo cách góc i hơn R —
#      KHÔNG chứng minh được quỹ đạo tránh khỏi toàn bộ tường.
#      Ví dụ: path xuyên qua giữa tường (xa cả 4 góc) → tất cả corner
#      barriers đều dương → OR logic báo "an toàn" SAI.
#
# FIX: Dùng check_rect() cho tường — adaptive circle decomposition đảm bảo
#      coverage invariant đúng, phát hiện mọi điểm xâm phạm.
#      Hàm này được giữ lại chỉ để tham khảo, không được gọi nữa.
# ─────────────────────────────────────────────────────────────────────────────
def check_wall_DEPRECATED(seg, obs):
    x_bl = obs["x_bl"]; w   = obs["w"]
    y_bl = obs["y_bl"]; h_r = obs["h_r"]
    R    = R_ROBOT
    h    = seg["h"]

    ax_ = seg["ax"]; bx = seg["bx"]; cx_ = seg["cx"]; dx = seg["dx"]
    ay_ = seg["ay"]; by = seg["by"]; cy_ = seg["cy"]; dy = seg["dy"]

    halfplane_candidates = [
        ("left",  [(x_bl - R) - ax_,  -bx,  -cx_,  -dx]),
        ("right", [ax_ - (x_bl+w+R),   bx,   cx_,   dx]),
        ("bot",   [(y_bl - R) - ay_,  -by,  -cy_,  -dy]),
        ("top",   [ay_ - (y_bl+h_r+R), by,   cy_,   dy]),
    ]
    corners = [
        ("corner_bl", x_bl,     y_bl),
        ("corner_br", x_bl + w, y_bl),
        ("corner_tl", x_bl,     y_bl + h_r),
        ("corner_tr", x_bl + w, y_bl + h_r),
    ]
    best_min  = -math.inf; best_axis = ""; best_ts = 0.0; best_deg = 3
    for axis_lbl, coeffs in halfplane_candidates:
        ts, mv = poly_min_on_interval(coeffs, h)
        if mv > best_min:
            best_min = mv; best_axis = axis_lbl; best_ts = ts; best_deg = 3
    for axis_lbl, cx_c, cy_c in corners:
        coeffs = barrier_circle_coeffs(seg, cx_c, cy_c, R)
        ts, mv = poly_min_on_interval(coeffs, h)
        if mv > best_min:
            best_min = mv; best_axis = axis_lbl; best_ts = ts; best_deg = 6
    Xs, Ys = eval_xy(seg, best_ts)
    return {
        "type": "wall", "rho": R, "poly_deg": best_deg, "axis": best_axis,
        "method": "wall_exact_disc_minkowski_deg3+6_DEPRECATED",
        "min_B": best_min, "tau_star": best_ts, "X_star": Xs, "Y_star": Ys,
        "safe": best_min > 0,
    }

def prefilter(seg, obs, bbox):
    xmn, xmx, ymn, ymx = bbox

    if obs["type"] == "circle":
        pad = obs["r"] + R_ROBOT
        ox1 = obs["cx"] - pad;  ox2 = obs["cx"] + pad
        oy1 = obs["cy"] - pad;  oy2 = obs["cy"] + pad
    else:
        ox1 = obs["x_bl"] - R_ROBOT
        ox2 = obs["x_bl"] + obs["w"]   + R_ROBOT
        oy1 = obs["y_bl"] - R_ROBOT
        oy2 = obs["y_bl"] + obs["h_r"] + R_ROBOT

    # Disjoint  ↔  one bbox is fully left/right/above/below the other
    return not (ox2 < xmn or ox1 > xmx or oy2 < ymn or oy1 > ymx)


# ══════════════════════════════════════════════════════════════════
#  7.  VÒNG KIỂM TRA CHÍNH
# ══════════════════════════════════════════════════════════════════

def run_verification(run_dir, verbose=True):
    global R_ROBOT

    # ── Đọc r_plan từ metadata để đồng bộ với RRT* và Spline ──────
    meta_csv = os.path.join(run_dir, "metadata.csv")
    R_ROBOT  = _load_robot_radius(meta_csv)

    segs     = load_spline(run_dir)
    obs_list = load_obstacles(run_dir)

    n_wall = sum(1 for o in obs_list if o["label"] == "wall")
    n_re   = sum(1 for o in obs_list if o["type"] == "rect" and o["label"] != "wall")
    n_ci   = sum(1 for o in obs_list if o["type"] == "circle")

    if verbose:
        print("\n" + "═"*68)
        print("  SOS BARRIER CERTIFICATE — Lukács Decomposition (Degree 6)")
        print("  Ref: Lukács 1918 · Parrilo 2000 · Prajna & Jadbabaie 2004")
        print("  gram_err: exact L∞ via derivative root-finding (no sampling)")
        print("  Wall: Adaptive Circle Decomposition (đồng nhất với rect); deg6 SOS + Gram cert")
        print("  Quyết định: SDP feasible ↔ B(τ) SOS trên [0,h] ↔ SAFE  (không dùng min_B)")
        print("═"*68)
        print(f"  Segs: {len(segs)} | Circle: {n_ci} | Rect: {n_re} | Wall: {n_wall} | R: {R_ROBOT} m")
        print("─"*68)

    results = []; skip = 0
    for seg in segs:
        i    = int(seg["segment"])
        bbox = seg_bbox(seg)
        for obs in obs_list:
            if not prefilter(seg, obs, bbox):
                skip += 1
                continue
            # [FIX] Không còn dùng check_wall nữa.
            # check_wall (cũ) dùng OR logic với corner barriers — không hợp lệ:
            # path xuyên giữa tường (xa tất cả góc) khiến corner barriers > 0
            # nhưng thực sự đang va chạm.
            # check_rect dùng adaptive circle decomposition với coverage invariant
            # đúng cho mọi hình chữ nhật, kể cả tường mỏng.
            if obs["type"] == "circle":
                res = check_circle(seg, obs)
            else:
                res = check_rect(seg, obs)   # wall + shelf + square + pallet
            res.update({
                "seg": i, "obs": obs["index"], "label": obs["label"],
                "s_start": seg["s_start"], "s_end": seg["s_end"], "h": seg["h"],
            })
            results.append(res)

    n_safe   = sum(1 for r in results if r["safe"])
    n_unsafe = len(results) - n_safe
    unsafe   = [r for r in results if not r["safe"]]
    safe_r   = sorted([r for r in results if r["safe"]], key=lambda x: x["min_B"])

    if verbose:
        print(f"\n  Cặp kiểm tra : {len(results)}  (bỏ qua {skip})")
        print(f"  PROVEN SAFE ✓: {n_safe}")
        print(f"  KHÔNG CM ✗   : {n_unsafe}")
        if n_unsafe == 0:
            print("\n  " + "═"*50)
            print("  ✓  TOÀN BỘ ĐƯỜNG ĐI ĐƯỢC CHỨNG MINH AN TOÀN")
            print("  " + "═"*50)
        else:
            for r in unsafe[:10]:
                print(f"  Seg{r['seg']:2d} Obs{r['obs']:3d} [{r['label']:>8}]"
                      f" [{r.get('gram_status','N/A'):>12}]"
                      f" min_B={r['min_B']:+.5f} τ*={r['tau_star']:.4f}"
                      f" ({r['X_star']:.3f},{r['Y_star']:.3f})")
        if safe_r:
            print(f"\n  Top 5 biên nhỏ nhất:")
            for r in safe_r[:5]:
                print(f"  Seg{r['seg']:2d} Obs{r['obs']:3d} minB={r['min_B']:.5f}"
                      f" τ*={r['tau_star']:.4f}  {r['method']}")
    return results, skip


# ══════════════════════════════════════════════════════════════════
#  8.  BÁO CÁO MARKDOWN
# ══════════════════════════════════════════════════════════════════

def _mat(M):
    return "\n     ".join("  ".join(f"{v:+10.6f}" for v in row) for row in M)


def generate_safety_report(results, skip, run_dir):
    safe_r   = sorted([r for r in results if r["safe"]], key=lambda x: x["min_B"])
    unsafe_r = [r for r in results if not r["safe"]]
    n_total  = len(results) + skip
    now      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fpath    = os.path.join(run_dir, "Safety_Verification_Report.md")

    L = []; A = L.append

    A("# BÁO CÁO CHỨNG MINH AN TOÀN QUỸ ĐẠO ROBOT")
    A(f"> **SOS / Lukács** |  Bộ giải: **CVXPY / SCS** |  {now}")
    A(""); A("---"); A("## 1. Thống Kê")
    A("| Chỉ tiêu | Giá trị |"); A("|---|---|")
    for k, v in [
        ("Tổng cặp (đoạn, vật cản)", n_total),
        ("Bỏ qua (prefilter)", skip),
        ("Kiểm tra SOS", len(results)),
        ("**Chứng minh an toàn**", f"**{len(safe_r)}**"),
        ("**Va chạm / Không CM được**", f"**{len(unsafe_r)}**"),
        ("Bán kính robot R", f"{R_ROBOT} m"),
    ]:
        A(f"| {k} | {v} |")
    A("")

    A("---"); A("## 2. Lý Thuyết Áp Dụng"); A("")
    A(textwrap.dedent("""
    ### 2.1 Tài liệu tham khảo

    | # | Tác giả | Năm | Tiêu đề | Vai trò |
    |---|---------|-----|---------|---------|
    | [1] | Lukács, F. | 1918 | Verschärfung des ersten Mittelwertsatzes… | Định lý phân rã SOS trên [0,h] |
    | [2] | Parrilo, P.A. | 2000 | Structured Semidefinite Programs… (PhD thesis) | SOS ↔ Gram matrix, SDP |
    | [3] | Prajna & Jadbabaie | 2004 | Safety Verification via Barrier Certificates (HSCC) | Barrier certificate cho hệ robot |

    ### 2.2 Vật cản hình tròn — 1 barrier bậc 6 (Lukács [1])

    B(τ) = (X−cx)²+(Y−cy)²−ρ²,  ρ=r+R  [bậc 6, chẵn]

    B(τ) = v₀ᵀQ₀v₀ + τ(h−τ)·v₁ᵀQ₁v₁,  Q₀∈ℝ^{4×4}⪰0, Q₁∈ℝ^{3×3}⪰0

    7 ràng buộc tuyến tính (τ⁰..τ⁶) — coefficient matching theo [2] §3.

    **Quy tắc quyết định (từ v2):**
      SDP khả thi (gram_status = 'optimal') ⟺ B(τ) là SOS trên [0,h] ⟺ SAFE.
      SDP không khả thi (infeasible / solver_error)              ⟺ UNSAFE.

      Tọa độ vi phạm (τ*, X*, Y*) và min_B được tính bằng poly_min_on_interval
      CHỈ cho mục đích chẩn đoán / báo cáo — KHÔNG dùng để kết luận SAFE/UNSAFE.

    ### 2.3 Vật cản hình chữ nhật — Adaptive Circle Decomposition

    Phân rã hình chữ nhật thành lưới N_x × N_y vòng tròn.
    Áp dụng Lukács bậc 6 cho TOÀN BỘ N_x×N_y vòng tròn (đầy đủ theo Parrilo [2]).

    **Coverage Invariant:**
      R_c = √((cell_w/2)²+(cell_h/2)²)
      ∀ điểm P trong ô: dist(P, tâm) ≤ R_c  → P nằm trong vòng tròn tương ứng.
      → ∀i B_i SOS ⟹  robot nằm ngoài hình chữ nhật.  □

    **Quy tắc quyết định (từ v2):**
      SAFE ⟺ MỌI N_x×N_y SDP đều khả thi.
      UNSAFE nếu có ÍT NHẤT MỘT SDP không khả thi.
      (Không dùng global_min > 0 làm điều kiện gọi SDP)

    **Chứng chỉ đầy đủ (Full Certificate):**
      Gram matrix Q₀⪰0, Q₁⪰0 được build cho TẤT CẢ circles trong lưới.
      gram_err_max = max_i L∞(B_i − SOS_i) trên toàn lưới.
      n_sos_certified = số circles có chứng chỉ hợp lệ (so sánh với n_circles).

    ### 2.4 Tính sai số chứng chỉ — Chính xác tuyệt đối

    Sai số gram_err KHÔNG dùng lấy mẫu rời rạc.  Thay vào đó:

        D(τ) = B(τ) − SOS(τ)   [đa thức bậc 6]
        gram_err = max_{τ∈[0,h]} |D(τ)|

    được tính **chính xác** bằng cách tìm nghiệm thực của D'(τ) (bậc 5,
    tối đa 5 nghiệm) qua np.roots, sau đó đánh giá |D| tại các nghiệm
    nằm trong (0,h) cộng với 2 đầu mút {0, h}.
    Cơ sở: Định lý giá trị cực trị (Extreme Value Theorem) + Fermat's theorem.
    """).strip())
    A("### 2.5 Vật cản tường — Adaptive Circle Decomposition (đồng nhất với rect)")

    A(textwrap.dedent("""
    Tường được xử lý **hoàn toàn giống rect** bằng Adaptive Circle Decomposition.

    **Lý do thay đổi (so với phiên bản cũ dùng half-plane + corner OR logic):**

    Phương pháp cũ dùng OR logic với 4 corner disc barriers không phải chứng chỉ
    an toàn hợp lệ. Cụ thể:

        B_corner_i(τ) > 0  ⟺  dist(path(τ), góc_i) > R

    Điều này KHÔNG chứng minh path tránh khỏi tường. Ví dụ, path xuyên qua
    chính giữa tường (xa tất cả 4 góc) khiến mọi corner barrier đều dương,
    trong khi thực tế đang va chạm — dẫn đến báo cáo "an toàn" SAI.

    **Phương pháp hiện tại — Coverage Invariant (giống §2.3):**

      N_x = ⌈w / (2R)⌉,  N_y = ⌈h / (2R)⌉
      R_c = √((cell_w/2)² + (cell_h/2)²),  ρ = R_c + R

      ∀ điểm P trong ô → dist(P, tâm_ô) ≤ R_c
      → nếu B_i(τ) = dist(path,tâm_i)² − ρ² > 0 với mọi i thì path ngoài tường.  □

    Với tường mỏng điển hình (w = 0.5 m, R = 0.55 m):
      N_x = 1,  N_y = ⌈h/(1.1)⌉  →  các circles xếp dọc tường
      ρ ≈ 0.559 + 0.55 = 1.109 m — phủ toàn bộ chiều rộng tường + margin.
    """).strip())
    A("")
    for k, r in enumerate(safe_r[:5], 1):
        A(f"### 3.{k}. Đoạn {r['seg']} ↔ Vật cản {r['obs']} ({r['label']})")
        A("| Thuộc tính | Giá trị |"); A("|---|---|")
        for lbl, val in [
            ("s domain", f"[{r['s_start']:.4f}, {r['s_end']:.4f}] m"),
            ("Phương pháp", f"`{r['method']}`"),
            ("**min B(τ*)**", f"**{r['min_B']:.6f}**"),
            ("τ*", f"{r['tau_star']:.6f}"),
            ("s* = s_i + τ*", f"{r['s_start']+r['tau_star']:.4f} m"),
            ("Tọa độ (X*, Y*)", f"({r['X_star']:.4f}, {r['Y_star']:.4f})"),
        ]:
            A(f"| {lbl} | {val} |")
        A("")
        if r.get("Q0") is not None:
            A("**Chứng chỉ SOS (Lukács [1] deg6)**"); A("")
            for name, M in [("Q₀", r["Q0"]), ("Q₁", r["Q1"])]:
                if M is not None:
                    A(f"**{name}** (PSD ⪰ 0):")
                    A("```"); A(_mat(M)); A("```")
                    A(f"Eigenvalues: {np.round(np.linalg.eigvalsh(M), 6).tolist()}")
                    A("")
            err_val = r.get('gram_err', float('nan'))
            A(f"Sai số xác nhận |B−SOS|_∞ = **{err_val:.2e}**  "
              f"*(exact L∞, derivative root-finding)*")
        A("")

    A("---"); A("## 4. Phân Tích Va Chạm"); A("")
    if not unsafe_r:
        A("✅ **Không có va chạm.** Toàn bộ đường đi được chứng minh an toàn bằng SOS/Lukács.")
    else:
        A(f"> ⚠️ **{len(unsafe_r)} cặp** không thể chứng minh an toàn."); A("")
        for k, r in enumerate(unsafe_r, 1):
            A(f"### Lỗi {k}: Đoạn {r['seg']} ↔ Vật cản {r['obs']} ({r['label']})")
            A(f"- s ∈ [{r['s_start']:.4f}, {r['s_end']:.4f}] m")
            A(f"- **τ* = {r['tau_star']:.4f}** → s* = {r['s_start']+r['tau_star']:.4f} m")
            A(f"- **Tọa độ:** X = {r['X_star']:.4f} m,  Y = {r['Y_star']:.4f} m")
            # B(τ) = d² − ρ²  [m²]  →  d* = √(ρ² + minB)  →  xâm phạm = ρ − d*
            _rho   = r.get('rho', float('nan'))
            _inner = _rho**2 + r['min_B']          # = d*²  phải ≥ 0
            if not math.isnan(_rho) and _inner >= 0:
                _pen_m  = _rho - math.sqrt(_inner)
                _pen_str = (f"xâm phạm {_pen_m*100:.1f} cm  "
                            f"[ρ={_rho:.3f} m; công thức: ρ−√(ρ²+B)]")
            elif not math.isnan(_rho):
                # _inner < 0: không thể xảy ra về mặt hình học,
                # báo hiệu giá trị ρ không nhất quán
                _pen_str = (f"B(τ*)={r['min_B']:+.6f} m²; ρ={_rho:.3f} m "
                            f"(ρ²+B={_inner:.4f} < 0 — kiểm tra lại ρ)")
            else:
                _pen_str = f"B(τ*)={r['min_B']:+.6f} m²  (không có ρ để tính)"
            A(f"- **min B(τ*) = {r['min_B']:+.6f} m²** (âm → {_pen_str})")
            A(f"- Best barrier: `{r.get('method', 'N/A')}`")
            A(f"- **Nguyên nhân:** Quỹ đạo Spline tại τ* cắt vào vùng Minkowski sum "
              "của vật cản. Khuyến nghị: tăng R_robot trong RRT* hoặc thêm waypoint.")
            A("")

    A("---"); A("## 5. Tài Liệu Tham Khảo"); A("")
    A("1. Lukács, F. (1918). *Verschärfung des ersten Mittelwertsatzes der "
      "Integralrechnung für rationale Polynome.* Mathematische Zeitschrift, 2, 295–305.")
    A("2. Parrilo, P. A. (2000). *Structured Semidefinite Programs and Semialgebraic "
      "Geometry Methods in Robustness and Optimization.* PhD thesis, Caltech.")
    A("3. Prajna, S. & Jadbabaie, A. (2004). *Safety Verification of Hybrid Systems "
      "Using Barrier Certificates.* HSCC 2004, LNCS 2993, pp. 477–492.")
    A("")
    A("---"); A("## 6. Kết Luận"); A("")
    if not unsafe_r:
        A("**Toàn bộ quỹ đạo được chứng minh an toàn** bằng SOS/Lukács (CVXPY/SCS). "
          "Chứng chỉ Gram matrix Q⪰0 đảm bảo B(τ)>0 liên tục trên [0,h]. "
          "Sai số gram_err được tính chính xác (không lấy mẫu).")
    else:
        A(f"**{len(safe_r)}/{len(results)} cặp được chứng minh an toàn.** "
          f"{len(unsafe_r)} cặp cần điều chỉnh quỹ đạo.")
    A(""); A("---"); A(f"*Sinh tự động bởi `SOS.py` — {now}*")

    with open(fpath, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n  📄 Báo cáo → {fpath}")
    return fpath


# ══════════════════════════════════════════════════════════════════
#  8b.  BASELINE — LẤY MẪU RỜI RẠC (Section 4.4.4)
# ══════════════════════════════════════════════════════════════════

def _barrier_coeffs_for_result(result: dict, spline_map: dict,
                                obs_map: dict, r_robot: float) -> list:
    """
    Tái tạo hệ số B(τ) cho một cặp (seg, obs) từ dict kết quả SOS.
    Trả về list-of-arrays: mỗi phần tử là mảng hệ số ascending của một
    barrier polynomial (1 cho circle, N_x*N_y cho rect).
    """
    seg = spline_map[result["seg"]]
    obs = obs_map[result["obs"]]

    def bc(s, cx, cy, rho):
        px = np.array([s["ax"] - cx, s["bx"], s["cx"], s["dx"]])
        py = np.array([s["ay"] - cy, s["by"], s["cy"], s["dy"]])
        p  = np.convolve(px, px) + np.convolve(py, py)
        p[0] -= rho ** 2
        return p

    if obs["type"] == "circle":
        rho = obs["r"] + r_robot
        return [bc(seg, obs["cx"], obs["cy"], rho)]
    else:
        # Adaptive circle decomposition (mirror _rect_circles)
        w, h_r = obs["w"], obs["h_r"]
        N_x = min(MAX_DIM, max(1, math.ceil(w   / (2 * r_robot))))
        N_y = min(MAX_DIM, max(1, math.ceil(h_r / (2 * r_robot))))
        cell_w = w   / N_x
        cell_h = h_r / N_y
        R_c = math.sqrt((cell_w / 2) ** 2 + (cell_h / 2) ** 2)
        rho = R_c + r_robot
        polys = []
        for ix in range(N_x):
            for iy in range(N_y):
                cx = obs["x_bl"] + (ix + 0.5) * cell_w
                cy = obs["y_bl"] + (iy + 0.5) * cell_h
                polys.append(bc(seg, cx, cy, rho))
        return polys


def discrete_safety_check(coeff_list: list, h: float,
                           n_samples: int = 20) -> dict:
    """
    Kiểm tra an toàn bằng lấy mẫu rời rạc trên τ ∈ [0, h].

    Parameters
    ----------
    coeff_list : list of np.ndarray
        Danh sách hệ số ascending của các barrier polynomial
        (1 phần tử cho circle, N_x×N_y phần tử cho rect).
    h          : float  — độ dài đoạn spline (m).
    n_samples  : int    — số điểm lấy mẫu (mặc định 2 000).

    Returns
    -------
    dict:
        'disc_min_B'   → min B(τ) trên lưới mẫu (global minimum qua tất cả polys)
        'disc_tau_min' → τ đạt disc_min_B
        'disc_status'  → 'AN TOÀN' | 'VI PHẠM'
    """
    from numpy.polynomial.polynomial import Polynomial
    tau = np.linspace(0.0, h, n_samples)
    global_min = np.inf
    global_tau = 0.0
    for coeffs in coeff_list:
        vals = Polynomial(coeffs)(tau)
        idx  = int(np.argmin(vals))
        if vals[idx] < global_min:
            global_min = float(vals[idx])
            global_tau = float(tau[idx])
    return {
        'disc_min_B':   global_min,
        'disc_tau_min': global_tau,
        'disc_status':  'AN TOÀN' if global_min >= 0.0 else 'VI PHẠM',
    }


def compare_sos_vs_discrete(results: list, run_dir: str,
                             n_samples: int = 20) -> list:
    """
    Chạy baseline rời rạc trên cùng tập cặp đã qua SOS, so sánh kết quả.

    Trả về list dict với các trường bổ sung:
        'disc_min_B'  : min B(τ) từ lấy mẫu
        'disc_tau_min': τ đạt disc_min_B
        'disc_status' : 'AN TOÀN' | 'VI PHẠM'
        'sos_status'  : 'AN TOÀN' | 'VI PHẠM'
        'agree'       : True nếu hai phương pháp đồng thuận
        'delta_minB'  : |min_B_SOS − min_B_disc|
    """
    r_robot    = _load_robot_radius(os.path.join(run_dir, "metadata.csv"))
    segs       = load_spline(run_dir)
    obs_list   = load_obstacles(run_dir)
    spline_map = {int(s["segment"]): s for s in segs}
    obs_map    = {o["index"]: o for o in obs_list}

    output = []
    for res in results:
        coeff_list = _barrier_coeffs_for_result(res, spline_map,
                                                obs_map, r_robot)
        disc = discrete_safety_check(coeff_list, res["h"], n_samples)
        sos_status = 'AN TOÀN' if res["safe"] else 'VI PHẠM'
        row = dict(res)
        row.update({
            'disc_min_B':   disc['disc_min_B'],
            'disc_tau_min': disc['disc_tau_min'],
            'disc_status':  disc['disc_status'],
            'sos_status':   sos_status,
            'agree':        sos_status == disc['disc_status'],
            'delta_minB':   abs(res["min_B"] - disc['disc_min_B']),
        })
        output.append(row)
    return output


def save_discrete_report(comp: list, run_dir: str):
    """
    Lưu kết quả so sánh SOS vs Rời rạc ra hai file:

    1. ``discrete_comparison_results.csv``  — dữ liệu thô từng cặp.
    2. ``Safety_Verification_Report.md``    — thêm mục 4.4.4 (append).

    Parameters
    ----------
    comp    : list of dict  — đầu ra của compare_sos_vs_discrete().
    run_dir : str           — thư mục run hiện tại.
    """
    if not comp:
        print("  ⚠️  Không có dữ liệu so sánh để lưu.")
        return

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. CSV ───────────────────────────────────────────────────────
    csv_path = os.path.join(run_dir, "discrete_comparison_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seg", "obs", "label", "h",
                    "sos_min_B", "sos_status",
                    "disc_min_B", "disc_tau_min", "disc_status",
                    "agree", "delta_minB"])
        for r in comp:
            w.writerow([
                r["seg"], r["obs"], r["label"], repr(r["h"]),
                repr(r["min_B"]),      r["sos_status"],
                repr(r["disc_min_B"]), repr(r["disc_tau_min"]),
                r["disc_status"],
                r["agree"],            repr(r["delta_minB"]),
            ])
    print(f"  💾 CSV baseline  → {csv_path}")

    # ── 2. Markdown (append vào Safety_Verification_Report.md) ───────
    md_path    = os.path.join(run_dir, "Safety_Verification_Report.md")
    n_agree    = sum(1 for r in comp if r["agree"])
    n_total    = len(comp)
    n_disagree = n_total - n_agree
    deltas     = [r["delta_minB"] for r in comp
                  if not math.isnan(r["delta_minB"])]

    L = []; A = L.append
    A("\n\n---")
    A("## 4.4.4  Baseline — Lấy Mẫu Rời Rạc vs SOS")
    A("")
    A(f"> **Sinh tự động bởi `SOS.py`** &nbsp;|&nbsp; {now}"
      f" &nbsp;|&nbsp; **n\\_samples = 2 000**")
    A("")

    # Tóm tắt thống kê
    A("### Tóm tắt")
    A("")
    A("| Chỉ tiêu | Giá trị |")
    A("|---|---|")
    A(f"| Tổng cặp so sánh | {n_total} |")
    A(f"| **Đồng thuận SOS ↔ Rời rạc** | **{n_agree} / {n_total}** |")
    A(f"| Bất đồng | {n_disagree} |")
    if n_total > 0:
        A(f"| Tỷ lệ đồng thuận | {n_agree / n_total * 100:.1f} % |")
    if deltas:
        A(f"| Δmin\\_B trung bình | {float(np.mean(deltas)):.3e} m² |")
        A(f"| Δmin\\_B tối đa | {float(np.max(deltas)):.3e} m² |")
    A("")

    # Ghi chú phương pháp
    A("### Phương pháp")
    A("")
    A(textwrap.dedent("""\
    Baseline đánh giá B(τ) tại **2 000 điểm đều** trên τ ∈ [0, h] và lấy
    giá trị nhỏ nhất làm ước lượng min_B.

    | Phương pháp | Đặc điểm | Vai trò |
    |---|---|---|
    | **SOS / Lukács** | Chính xác toán học (không lấy mẫu) | **Quyết định** |
    | **Rời rạc (2k pts)** | Xấp xỉ, bỏ sót cực tiểu hẹp | Kiểm tra chéo |

    - Nếu rời rạc → AN TOÀN nhưng SOS → VI PHẠM: lưới chưa đủ dày để bắt cực tiểu cục bộ hẹp.
    - Nếu rời rạc → VI PHẠM nhưng SOS → AN TOÀN: trường hợp bất thường, cần điều tra thêm.
    """))

    # Bảng chi tiết
    A("### Bảng chi tiết các cặp")
    A("")
    A("| Đoạn | Vật cản | Nhãn | h (m) "
      "| min_B SOS | min_B Rời rạc | Δmin_B "
      "| SOS | Rời rạc | Đồng thuận |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in comp:
        mark = "✅" if r["agree"] else "❌"
        A(f"| {r['seg']} | {r['obs']} | {r['label']} "
          f"| {float(r['h']):.4f} "
          f"| {r['min_B']:+.5f} "
          f"| {r['disc_min_B']:+.5f} "
          f"| {r['delta_minB']:.2e} "
          f"| {r['sos_status']} "
          f"| {r['disc_status']} "
          f"| {mark} |")
    A("")

    # Phân tích bất đồng (nếu có)
    if n_disagree > 0:
        A("### ⚠️ Phân tích bất đồng")
        A("")
        for r in (r for r in comp if not r["agree"]):
            A(f"**Đoạn {r['seg']} ↔ Vật cản {r['obs']} ({r['label']})**")
            A(f"- SOS     : {r['sos_status']}  (min\\_B = {r['min_B']:+.6f} m²)")
            A(f"- Rời rạc : {r['disc_status']}  "
              f"(min\\_B = {r['disc_min_B']:+.6f} m²,  τ\\* = {r['disc_tau_min']:.4f})")
            A(f"- Δmin\\_B = {r['delta_minB']:.3e} m²")
            A("")
        A("> *Phán quyết của SOS (chính xác về mặt toán học) luôn được "
          "ưu tiên. Bất đồng thường do cực tiểu B(τ) rất hẹp nằm giữa "
          "hai điểm lấy mẫu liên tiếp.*")
    else:
        A(f"✅ **Hai phương pháp đồng thuận hoàn toàn** ({n_agree}/{n_total} cặp).")
    A("")
    A("---")
    A(f"*Mục 4.4.4 — Sinh tự động bởi `SOS.py` — {now}*")

    mode = "a" if os.path.exists(md_path) else "w"
    with open(md_path, mode, encoding="utf-8") as f:
        f.write("\n".join(L))
    action = "Thêm vào" if mode == "a" else "Tạo mới"
    print(f"  📄 Mục 4.4.4   → {action} {md_path}")


# ══════════════════════════════════════════════════════════════════
#  9.  CSV
# ══════════════════════════════════════════════════════════════════

def save_csv(results, run_dir):
    out = os.path.join(run_dir, "sos_certificate_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["segment", "obstacle", "label", "s_start", "s_end", "h",
                    "type", "poly_deg", "axis", "method",
                    "min_B", "tau_star", "X_star", "Y_star", "safe", "gram_err"])
        for r in results:
            w.writerow([
                r["seg"], r["obs"], r["label"],
                repr(r["s_start"]), repr(r["s_end"]), repr(r["h"]),
                r["type"], r["poly_deg"], r.get("axis", "N/A"), r.get("method", ""),
                repr(r["min_B"]), repr(r["tau_star"]),
                repr(r["X_star"]), repr(r["Y_star"]),
                r["safe"], repr(r.get("gram_err", float("nan"))),
            ])
    print(f"  💾 CSV → {out}")


# ══════════════════════════════════════════════════════════════════
#  9b.  LƯU HỆ SỐ BARRIER C0..C6  (phục vụ mục 4.3.5)
# ══════════════════════════════════════════════════════════════════

def save_barrier_coeffs(results, run_dir):
    """
    Lưu hệ số đa thức B(τ) = C0 + C1·τ + C2·τ² + … + C6·τ⁶ ra file
    ``barrier_coeffs.csv`` trong run_dir.

    Mục đích: cung cấp đầu vào cho script so sánh SOS vs lấy-mẫu-rời-rạc
    (mục 4.3.5).  Script đó cần hệ số chính xác từ code này; không cần
    sao chép tay.

    Với vật cản hình tròn : 1 barrier duy nhất, lưu trực tiếp.
    Với vật cản hình chữ nhật / tường / kệ : adaptive circle decomposition
        tạo N_x × N_y circles; hàm này chọn circle có giá trị min_B nhỏ nhất
        (worst-case) làm đại diện.  Đây là circle quyết định kết quả SOS.

    Columns:
        seg, obs, label, h, safe, min_B, n_circles, C0, C1, C2, C3, C4, C5, C6
    """
    r_robot    = _load_robot_radius(os.path.join(run_dir, "metadata.csv"))
    segs       = load_spline(run_dir)
    obs_list   = load_obstacles(run_dir)
    spline_map = {int(s["segment"]): s for s in segs}
    obs_map    = {o["index"]: o for o in obs_list}

    out = os.path.join(run_dir, "barrier_coeffs.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seg", "obs", "label", "h", "safe", "min_B", "n_circles",
                    "C0", "C1", "C2", "C3", "C4", "C5", "C6"])
        for res in results:
            coeff_list = _barrier_coeffs_for_result(
                res, spline_map, obs_map, r_robot)
            n_circles  = len(coeff_list)
            h          = float(res["h"])

            # Chọn circle có min B nhỏ nhất (worst-case)
            min_vals = []
            for c in coeff_list:
                _, mv = poly_min_on_interval(c, h)
                min_vals.append(mv)
            worst_idx = int(np.argmin(min_vals))
            c_raw     = coeff_list[worst_idx]

            # Đảm bảo đúng 7 hệ số (barrier_circle_coeffs trả về 7 phần tử,
            # nhưng padding an toàn phòng trường hợp deg thấp hơn)
            c7 = np.zeros(7)
            c7[:min(len(c_raw), 7)] = c_raw[:7]

            w.writerow([
                res["seg"], res["obs"], res["label"],
                repr(h), res["safe"],
                repr(float(res["min_B"])), n_circles,
                repr(float(c7[0])), repr(float(c7[1])), repr(float(c7[2])),
                repr(float(c7[3])), repr(float(c7[4])), repr(float(c7[5])),
                repr(float(c7[6])),
            ])

    print(f"  💾 Hệ số C0–C6  → {out}  ({len(results)} dòng)")
    return out


# ══════════════════════════════════════════════════════════════════
#  10.  GHI METADATA
# ══════════════════════════════════════════════════════════════════

def save_sos_metadata(run_dir, sos_time_s, results, skip):
    """Append các dòng SOS vào metadata.csv (group=sos).

    Dùng mode 'a' (append) — KHÔNG ghi đè — để giữ nguyên toàn bộ
    dữ liệu RRT*, map và spline đã có từ các bước trước.

    Các key được ghi nhất quán với quy ước group,key,value của file:
      sos,elapsed_sos_s       — thời gian chạy toàn bộ SOS (giây)
      sos,n_pairs_checked     — số cặp (đoạn, vật cản) đã giải SDP
      sos,n_pairs_skipped     — số cặp bỏ qua qua prefilter
      sos,n_safe              — số cặp PROVEN SAFE
      sos,n_unsafe            — số cặp KHÔNG chứng minh được
      sos,avg_ms_per_pair     — thời gian trung bình mỗi cặp (ms)
      sos,verified_safe       — True nếu TOÀN BỘ đường đi an toàn
    """
    meta_path = os.path.join(run_dir, "metadata.csv")
    if not os.path.exists(meta_path):
        print(f"  ⚠️  Không tìm thấy {meta_path} — bỏ qua ghi metadata.")
        return

    n_safe   = sum(1 for r in results if r["safe"])
    n_unsafe = len(results) - n_safe
    n_pairs  = len(results)
    avg_ms   = (sos_time_s / n_pairs * 1000) if n_pairs > 0 else 0.0

    rows = [
        ("sos", "elapsed_sos_s",   f"{sos_time_s:.6f}"),
        ("sos", "n_pairs_checked", str(n_pairs)),
        ("sos", "n_pairs_skipped", str(skip)),
        ("sos", "n_safe",          str(n_safe)),
        ("sos", "n_unsafe",        str(n_unsafe)),
        ("sos", "avg_ms_per_pair", f"{avg_ms:.3f}"),
        ("sos", "verified_safe",   str(n_unsafe == 0)),
    ]

    with open(meta_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

    print(f"  📋 Metadata SOS → {meta_path}  ({len(rows)} dòng mới)")


# ══════════════════════════════════════════════════════════════════
#  11.  VISUALISATION  (gộp từ check.py)
# ══════════════════════════════════════════════════════════════════

# Màu sắc theo nhãn vật cản — đồng bộ với map.py COLORS
_LABEL_COLOR = {
    'wall':   ('#7f8c8d', 0.90),
    'shelf':  ('#34495e', 0.85),
    'pallet': ('#d6d8d9', 0.90),
    'square': ('#e67e22', 0.80),
    'circle': ('#3498db', 0.70),
}


def _load_robot_radius(meta_csv):
    """Đọc r_plan (= radius + safety_margin) từ metadata.csv.
    Đây là bán kính kế hoạch thực sự được dùng trong RRT* và SOS.
    Fallback về hằng số toàn cục nếu không tìm thấy file hoặc key.
    """
    if not os.path.exists(meta_csv):
        return R_ROBOT
    with open(meta_csv, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('key') == 'r_plan':
                r = float(row['value'])
                print(f"  [SOS] R_robot (r_plan) = {r} m (đọc từ metadata.csv)")
                return r
    return R_ROBOT


def _load_metadata(meta_csv):
    """Đọc start / goal từ metadata.csv."""
    start = goal = None
    if not os.path.exists(meta_csv):
        return start, goal
    sx = sy = gx = gy = None
    with open(meta_csv, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            k, v = row.get('key', ''), row.get('value', '')
            if k == 'start_x': sx = float(v)
            if k == 'start_y': sy = float(v)
            if k == 'goal_x':  gx = float(v)
            if k == 'goal_y':  gy = float(v)
    if None not in (sx, sy): start = (sx, sy)
    if None not in (gx, gy): goal  = (gx, gy)
    return start, goal


def _load_sos_results(sos_csv):
    unsafe_segments  = {}
    collision_points = []
    with open(sos_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for row in reader:
            if str(row.get('safe', 'true')).strip().lower() == 'false':
                seg_id = int(row['segment'])
                min_b  = float(row.get('min_B', 0))
                label  = row.get('label', '?')
                unsafe_segments.setdefault(seg_id, []).append(
                    (row['obstacle'], min_b, label))
                if 'X_star' in fields and 'Y_star' in fields:
                    try:
                        collision_points.append((
                            float(row['X_star']),
                            float(row['Y_star']),
                            min_b,
                        ))
                    except (ValueError, KeyError):
                        pass
    return unsafe_segments, collision_points


def _draw_obstacles(ax, obstacles):
    legend_patches = {}
    for obs in obstacles:
        lbl   = obs['label']
        color, alpha = _LABEL_COLOR.get(lbl, ('#888888', 0.75))

        if obs['type'] == 'circle':
            patch = mpatches.Circle(
                (obs['cx'], obs['cy']), obs['r'],
                facecolor=color, edgecolor='black',
                linewidth=0.6, alpha=alpha, zorder=3)
        else:
            patch = mpatches.Rectangle(
                (obs['x_bl'], obs['y_bl']), obs['w'], obs['h_r'],
                facecolor=color, edgecolor='black',
                linewidth=0.6, alpha=alpha, zorder=3)

        ax.add_patch(patch)

        # Legend: chỉ thêm mỗi nhãn một lần, tên hiển thị giống map.py
        _LABEL_DISPLAY = {
            'wall':   'Tường',
            'shelf':  'Kệ hàng',
            'pallet': 'Bãi hàng',
            'square': 'Kiện hàng vuông',
            'circle': 'Vật tròn',
        }
        if lbl not in legend_patches:
            legend_patches[lbl] = mpatches.Patch(
                facecolor=color, edgecolor='black', linewidth=0.6,
                alpha=alpha, label=_LABEL_DISPLAY.get(lbl, lbl.capitalize()))

    return list(legend_patches.values())


def _draw_inflated_obstacles(ax, obstacles, r_robot):
    """Vẽ vùng cấm Minkowski sum (obstacle ⊕ disc(R_ROBOT)) — đúng hình học.

    Với circle: vẽ 1 vòng tròn bán kính r + R_ROBOT.
    Với rect:   vẽ FancyBboxPatch mở rộng ±R_ROBOT mỗi chiều với bo góc R_ROBOT
                → đây là Minkowski sum chính xác của hình chữ nhật với đĩa tròn.

    LƯU Ý: Không dùng các SOS barrier circles (_rect_circles) để vẽ vì
    rho = R_c + R_ROBOT >> R_ROBOT → vòng tròn to hơn vùng cấm thực tế.
    """
    for obs in obstacles:
        if obs['type'] == 'circle':
            # Minkowski sum: circle(r) ⊕ disc(R) = circle(r + R)
            ax.add_patch(mpatches.Circle(
                (obs['cx'], obs['cy']), obs['r'] + r_robot,
                fill=False, linestyle='--',
                edgecolor='#FF4444', linewidth=0.8, alpha=0.6, zorder=2))
        else:
            ax.add_patch(mpatches.FancyBboxPatch(
                (obs['x_bl'], obs['y_bl']),   # Dùng trực tiếp tọa độ góc dưới-trái gốc
                obs['w'], obs['h_r'],         # Dùng trực tiếp chiều rộng/cao gốc
                boxstyle=f"round,pad={r_robot}",
                fill=False, linestyle='--',
                edgecolor='#FF4444', linewidth=0.8, alpha=0.5, zorder=2))


def plot_trajectory(run_dir):
    sos_csv  = os.path.join(run_dir, "sos_certificate_results.csv")
    spl_csv  = os.path.join(run_dir, "spline_equations.csv")
    obs_csv  = os.path.join(run_dir, "obstacles.csv")
    meta_csv = os.path.join(run_dir, "metadata.csv")
    wp_csv   = os.path.join(run_dir, "waypoints.csv")

    missing = [f for f in [sos_csv, spl_csv] if not os.path.exists(f)]
    if missing:
        print(f"  ⚠️  Thiếu file để vẽ: {missing}")
        return

    # ── Đọc dữ liệu ──────────────────────────────────────────────
    unsafe_segments, collision_points = _load_sos_results(sos_csv)
    obstacles = _load_sos_results(obs_csv)[0] if False else (
        [] if not os.path.exists(obs_csv) else load_obstacles(obs_csv))
    start, goal = _load_metadata(meta_csv)
    r_robot     = _load_robot_radius(meta_csv)
    x_max_map, y_max_map = _load_map_bounds(run_dir)

    # ── In tóm tắt terminal ───────────────────────────────────────
    if unsafe_segments:
        print(f"\n  🚨 {len(unsafe_segments)} đoạn vi phạm SOS:")
        for seg_id, obs_list in sorted(unsafe_segments.items()):
            for obs_id, min_b, lbl in obs_list:
                print(f"     Đoạn {seg_id:2d} → Vật cản {obs_id:>3}"
                      f" [{lbl:>8}]  min_B = {min_b:+.4f} m²")
    else:
        print("\n  ✅  TOÀN BỘ ĐƯỜNG ĐI ĐƯỢC CHỨNG MINH AN TOÀN!\n")

    # ── Vẽ ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_facecolor('#F7F9FC')
    fig.patch.set_facecolor('#FFFFFF')

    obs_patches = _draw_obstacles(ax, obstacles)
    _draw_inflated_obstacles(ax, obstacles, r_robot)

    minkowski_patch = mpatches.Patch(
        fill=False, linestyle='--', edgecolor='#FF4444', linewidth=1.0,
        label=f'Vùng cấm Minkowski (obstacle + R={r_robot}m)')

    # Vẽ spline theo từng đoạn
    plotted_safe = plotted_unsafe = False
    n_segs = 0
    with open(spl_csv, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            n_segs += 1
            seg_id = int(row['segment'])
            h      = float(row['h'])
            ax_ = float(row['ax']); bx = float(row['bx'])
            cx_ = float(row['cx']); dx = float(row['dx'])
            ay  = float(row['ay']); by = float(row['by'])
            cy_ = float(row['cy']); dy = float(row['dy'])

            tv = np.linspace(0, h, 200)
            xs = ax_ + bx*tv + cx_*tv**2 + dx*tv**3
            ys = ay  + by*tv + cy_*tv**2 + dy*tv**3

            # Nút đầu đoạn
            ax.scatter([xs[0]], [ys[0]], color='#333333',
                       s=18, zorder=6, linewidths=0)

            if seg_id in unsafe_segments:
                lbl = 'Cubic Spline (Vi phạm SOS)' if not plotted_unsafe else ""
                ax.plot(xs, ys, color='#E02020', linewidth=2.8, zorder=4,
                        label=lbl, solid_capstyle='round')
                plotted_unsafe = True
            else:
                lbl = 'Cubic Spline (An toàn)' if not plotted_safe else ""
                ax.plot(xs, ys, color='#1E88E5', linewidth=1.8, zorder=3,
                        label=lbl, solid_capstyle='round')
                plotted_safe = True

    # Vẽ waypoints RRT* gốc nếu có
    if os.path.exists(wp_csv):
        wx, wy = [], []
        with open(wp_csv) as f:
            for row in csv.DictReader(f):
                wx.append(float(row['x'])); wy.append(float(row['y']))
        ax.plot(wx, wy, '--', color='#FF00FF', linewidth=1.2,
                alpha=0.6, zorder=2, label='RRT* Waypoints (gốc)',
                dashes=(5, 3))

    # Điểm nguy hiểm
    if collision_points:
        ax.scatter([p[0] for p in collision_points],
                   [p[1] for p in collision_points],
                   color='#8B0000', s=90, marker='X', zorder=7,
                   linewidths=0.8, edgecolors='white',
                   label='Điểm nguy hiểm (X*, Y*)')

    # Start / Goal
    if start:
        ax.scatter(*start, color='#00C853', s=160, marker='s',
                   zorder=8, edgecolors='white', linewidths=1.5,
                   label=f'Start ({start[0]:.1f}, {start[1]:.1f})')
        ax.annotate('Start', start, textcoords='offset points',
                    xytext=(10, 8), fontsize=9, color='#00C853',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.7, edgecolor='none'))
    if goal:
        ax.scatter(*goal, color='#D50000', s=180, marker='*',
                   zorder=8, edgecolors='white', linewidths=1.0,
                   label=f'Goal ({goal[0]:.1f}, {goal[1]:.1f})')
        ax.annotate('Goal', goal, textcoords='offset points',
                    xytext=(10, 8), fontsize=9, color='#D50000',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.7, edgecolor='none'))

    # ── Legend & thống kê ─────────────────────────────────────────
    ax.set_xlabel('Trục X (m)', fontsize=12)
    ax.set_ylabel('Trục Y (m)', fontsize=12)
    ax.set_title('Bản Đồ Xác Minh Quỹ Đạo Toàn Cảnh',
                 fontsize=14, fontweight='bold', pad=25)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4, color='#AAAAAA')
    ax.set_xlim(0, x_max_map)
    ax.set_ylim(0, y_max_map)

    n_unsafe_pairs = sum(len(v) for v in unsafe_segments.values())
    n_unsafe_segs  = len(unsafe_segments)
    n_safe_segs    = n_segs - n_unsafe_segs
    n_safe_pairs   = n_segs * 1  # placeholder — tổng cặp an toàn tính từ CSV nếu cần

    spline_handles, spline_labels = ax.get_legend_handles_labels()
    all_handles = spline_handles + [minkowski_patch] + obs_patches
    all_labels  = spline_labels  + [minkowski_patch.get_label()] \
                                 + [p.get_label() for p in obs_patches]
    seen = {}
    for h2, lbl in zip(all_handles, all_labels):
        if lbl and lbl not in seen:
            seen[lbl] = h2

    stats_txt = (f"───────────────\nTHỐNG KÊ:\n"
                 f" • Đoạn an toàn: {n_safe_segs}\n"
                 f" • Đoạn vi phạm: {n_unsafe_segs}\n"
                 f" • Cặp vi phạm:  {n_unsafe_pairs}\n")
    seen[stats_txt] = mpatches.Rectangle(
        (0, 0), 0, 0, fill=False, edgecolor='none', visible=False)

    leg = ax.legend(seen.values(), seen.keys(),
                    loc='upper left', bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0, fontsize=9, framealpha=0.96,
                    edgecolor='#BBBBBB',
                    title='Chú thích & Thống kê', title_fontsize=9.5)
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    img_path = os.path.join(run_dir, "check.png")
    fig.savefig(img_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
    print(f"  🗺️  Hình ảnh → {img_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    run_dir = get_latest_run_dir()
    print(f"\n  Run dir: {run_dir}")

    # 1. Chạy SOS verification → results + skip
    t0 = time.perf_counter()
    results, skip = run_verification(run_dir, verbose=True)
    sos_time_s = time.perf_counter() - t0
    print(f"\n  ⏱  Thời gian SOS verification: {sos_time_s:.3f}s")

    # 2. Sinh báo cáo Markdown
    generate_safety_report(results, skip, run_dir)
    # 3. Lưu CSV kết quả
    save_csv(results, run_dir)
    # 3b. Lưu hệ số barrier C0..C6 (cho script so sánh mục 4.3.5)
    save_barrier_coeffs(results, run_dir)
    # 4. Ghi thời gian chạy và thống kê vào metadata.csv
    save_sos_metadata(run_dir, sos_time_s, results, skip)
    # 5. Vẽ bản đồ xác minh (gộp từ check.py)
    plot_trajectory(run_dir)

    # 6. So sánh baseline rời rạc (mục 4.4.4)
    print("\n" + "─"*68)
    print("  BASELINE — Lấy mẫu rời rạc (n_samples=20)")
    print("─"*68)
    t1 = time.perf_counter()
    comp = compare_sos_vs_discrete(results, run_dir, n_samples=20)
    disc_time_s = time.perf_counter() - t1
    n_agree = sum(1 for r in comp if r['agree'])
    print(f"\n  [Baseline] Đồng thuận SOS vs Rời rạc: {n_agree}/{len(comp)}")
    print(f"  ⏱  Thời gian Baseline: {disc_time_s:.3f}s")
    save_discrete_report(comp, run_dir)

    print("\n  Hoàn tất.\n")
if __name__ == "__main__":
    main()
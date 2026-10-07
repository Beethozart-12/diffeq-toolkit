"""diffeq-toolkit 的网页界面：在网页输入 LaTeX 公式，运行时自动转换为
Python 可执行的代码（sympy -> lambdify），再交给 diffeq 求解/计算。

运行方式（在仓库根目录执行）：
    python web/app.py [--port 8000] [--host 127.0.0.1]

然后浏览器打开 http://127.0.0.1:8000/

设计要点
--------
- 用标准库 http.server，无需 Flask 等额外 Web 框架（除了 [web] 里的 sympy/antlr4）。
- 公式解析：sympy.parsing.latex.parse_latex 把 LaTeX 转成 sympy 表达式；
  再用 sympy.lambdify 生成「Python 读得懂」的数值函数，交给 diffeq.ode.solve_ivp
  等进行数值计算。
- 这样用户输入的 LaTeX 在服务器端被「翻译」成真正的 Python 函数并运行，
  前端会同时回显该 Python 源码，直观展示「运行时自动用 python 读得懂的方式运算」。
"""

import os
import sys
import json
import re
import argparse
import io
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# 让本文件（web/app.py）所在的仓库根目录进入 sys.path，从而能 `import diffeq`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

WEB_DIR = os.path.dirname(os.path.abspath(__file__))

# 必须在 import matplotlib 之前设定配置目录（无头沙箱里 HOME/.matplotlib 不可写）。
os.environ.setdefault("MPLCONFIGDIR", os.path.join(WEB_DIR, ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import sympy as sp
from sympy.parsing.latex import parse_latex
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头环境，必须早于 pyplot 导入
import matplotlib.pyplot as plt
from diffeq.ode import solve_ivp
from diffeq.pde import solve_heat_nd, solve_wave_nd, solve_laplace_nd


# --------------------------------------------------------------------------- #
# 核心：LaTeX -> sympy -> Python 可执行函数
# --------------------------------------------------------------------------- #
def latex_to_python(latex: str):
    """把一段 LaTeX 解析为 (sympy_expr, free_symbols_set)。

    返回表达式本身（sympy 对象）以及其中出现的自由符号名集合。
    """
    latex = (latex or "").strip()
    if not latex:
        raise ValueError("LaTeX 公式为空")
    expr = parse_latex(latex)
    # sympy 的 LaTeX 解析器会把 \pi 解析成普通符号 Symbol('pi')，
    # 统一替换为圆周率 sp.pi（否则 lambdify 后无法求值）。
    expr = expr.subs(sp.Symbol("pi"), sp.pi)
    syms = {str(s) for s in expr.free_symbols}
    return expr, syms


def build_ode_function(latex: str, indep: str = None, dep: str = None):
    """解析一个一阶常微分方程（形如 \\frac{dy}{dt} = f(t, y)）。

    返回 (callable f(indep, dep), indep, dep, rhs_sympy)。
    若输入不含 `=`，则把整段当作右侧 f 处理（默认变量 t, y）。
    """
    expr, _ = latex_to_python(latex)

    if isinstance(expr, sp.Equality):
        lhs, rhs = expr.lhs, expr.rhs
    else:
        lhs, rhs = None, expr

    if lhs is not None and isinstance(lhs, sp.Derivative):
        # 注意：sympy 把一阶导数表示为 Derivative(y, (t, 1))，
        # 求导变量是 (var, order) 元组，需用 .variables 提取真正的自变量。
        d = lhs.args
        dep = str(d[0])
        indep = str(lhs.variables[0])
    else:
        # 用户只给了右侧表达式，使用默认/指定变量
        indep = indep or "t"
        dep = dep or "y"

    if indep == dep:
        raise ValueError("自变量与因变量不能相同")

    # 把自由符号 e 当作自然常数 e（Euler 数）处理；若 e 恰好被用作自变量/
    # 因变量名则跳过，避免误替换。
    if "e" not in (indep, dep):
        rhs = rhs.subs(sp.Symbol("e"), sp.E)

    f = sp.lambdify((indep, dep), rhs, modules="numpy")
    return f, indep, dep, rhs


def expr_python_src(rhs, indep, dep):
    """生成可读的 Python lambda 源码字符串，用于前端展示。"""
    body = sp.printing.lambdarepr.lambdarepr(rhs)
    return f"lambda {indep}, {dep}: {body}"


def fig_to_base64(fig) -> str:
    """把 matplotlib Figure 编码为 base64 PNG data URI。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# --------------------------------------------------------------------------- #
# API 处理
# --------------------------------------------------------------------------- #
def api_solve(payload: dict) -> dict:
    """ODE 求解：LaTeX 方程 -> 解析 -> solve_ivp -> 结果 + 图。"""
    latex = payload.get("latex", "")
    try:
        y0 = float(payload.get("y0", 1.0))
        t0 = float(payload.get("t0", 0.0))
        tf = float(payload.get("tf", 2.0))
        if tf <= t0:
            raise ValueError("t 终值必须大于初值")
    except (TypeError, ValueError) as e:
        return {"ok": False, "error": f"参数错误：{e}"}

    try:
        f, indep, dep, rhs = build_ode_function(latex)
    except Exception as e:
        return {"ok": False, "error": f"公式解析失败：{e}"}

    try:
        t, y = solve_ivp(f, y0, (t0, tf))
    except Exception as e:
        return {"ok": False, "error": f"求解失败：{e}"}

    # 绘图：t - y
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(np.asarray(t), np.asarray(y), color="#2c7be5", lw=2)
    ax.set_xlabel(f"{indep}")
    ax.set_ylabel(f"{dep}({indep})")
    ax.set_title(f"数值解：d{dep}/d{indep} = {sp.latex(rhs)}")
    ax.grid(True, alpha=0.3)
    plot = fig_to_base64(fig)

    # 抽稀返回：最多 200 个点，避免响应体积过大
    n = len(t)
    step = max(1, n // 200)
    return {
        "ok": True,
        "mode": "ode",
        "indep": indep,
        "dep": dep,
        "expr_sympy": sp.pretty(rhs),
        "expr_latex": sp.latex(rhs),
        "python_src": expr_python_src(rhs, indep, dep),
        "n_points": n,
        "t": [float(v) for v in t[::step]],
        "y": [float(v) for v in y[::step]],
        "plot": plot,
    }


def api_eval(payload: dict) -> dict:
    """通用表达式求值/绘图：LaTeX 表达式 -> 解析 -> 在 [a, b] 上求值。"""
    latex = payload.get("latex", "")
    var = (payload.get("var") or "").strip() or None
    try:
        a = float(payload.get("a", -5.0))
        b = float(payload.get("b", 5.0))
        n = int(payload.get("n", 400))
        if b <= a:
            raise ValueError("区间上界必须大于下界")
        if n <= 0:
            raise ValueError("点数必须为正整数")
    except (TypeError, ValueError) as e:
        return {"ok": False, "error": f"参数错误：{e}"}

    try:
        expr, syms = latex_to_python(latex)
    except Exception as e:
        return {"ok": False, "error": f"公式解析失败：{e}"}

    # 确定自变量
    if var is None:
        if len(syms) == 1:
            var = next(iter(syms))
        elif "x" in syms:
            var = "x"
        else:
            return {
                "ok": False,
                "error": f"无法确定自变量，请指定 var（表达式中的符号：{sorted(syms)}）",
            }
    if var not in syms:
        return {"ok": False, "error": f"指定的自变量 '{var}' 不在表达式中（可用：{sorted(syms)}）"}

    # 把自由符号 e 当作自然常数 e（Euler 数）处理；若 e 被用作自变量则跳过。
    if "e" != var:
        expr = expr.subs(sp.Symbol("e"), sp.E)

    try:
        g = sp.lambdify(var, expr, modules="numpy")
        xs = np.linspace(a, b, n)
        ys = np.asarray(g(xs), dtype=float)
    except Exception as e:
        return {"ok": False, "error": f"求值失败：{e}"}

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(xs, ys, color="#e8532c", lw=2)
    ax.set_xlabel(var)
    ax.set_ylabel(f"f({var})")
    ax.set_title(f"f({var}) = {sp.latex(expr)}")
    ax.grid(True, alpha=0.3)
    plot = fig_to_base64(fig)

    # 返回若干采样点
    sample_idx = np.linspace(0, n - 1, min(12, n)).astype(int)
    return {
        "ok": True,
        "mode": "eval",
        "var": var,
        "expr_sympy": sp.pretty(expr),
        "expr_latex": sp.latex(expr),
        "python_src": f"lambda {var}: {sp.printing.lambdarepr.lambdarepr(expr)}",
        "x": [float(xs[i]) for i in sample_idx],
        "y": [float(ys[i]) for i in sample_idx],
        "plot": plot,
    }


# --------------------------------------------------------------------------- #
# PDE 求解：n 维热传导 / 波动 / Laplace
# --------------------------------------------------------------------------- #
PDE_VAR_NAMES = {1: ["x"], 2: ["x", "y"], 3: ["x", "y", "z"]}
PDE_GRID_CAPS = {1: 801, 2: 301, 3: 81}          # 每维网格点数上限（控内存）
PDE_STEP_CAPS = {1: 20000, 2: 3000, 3: 800}      # 演化步数上限（控耗时）


def _pde_num(payload, key, default, lo=None, hi=None, what=""):
    try:
        v = float(payload.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"参数 {key} 不是数字")
    if lo is not None and v < lo:
        raise ValueError(f"参数 {what or key} 必须 ≥ {lo}")
    if hi is not None and v > hi:
        raise ValueError(f"参数 {what or key} 必须 ≤ {hi}")
    return v


def _latex_field(latex, var_names, grids, what):
    """把 LaTeX 表达式解析成给定网格上的实数场（广播标量）。"""
    expr, syms = latex_to_python(latex)
    unknown = sorted(syms - set(var_names))
    if unknown:
        if "nabla" in unknown or "Delta" in unknown:
            raise ValueError(
                f"{what}不支持 ∇ 算子：\\nabla / \\Delta 只能出现在「方程」输入框中"
            )
        raise ValueError(f"{what}中出现了未知符号 {unknown}（可用变量：{var_names}）")
    if "e" not in var_names:
        expr = expr.subs(sp.Symbol("e"), sp.E)
    g = sp.lambdify(var_names, expr, modules="numpy")
    arr = np.asarray(g(*grids), dtype=float)
    if arr.shape != grids[0].shape:
        arr = np.broadcast_to(arr, grids[0].shape).astype(float)
    return arr


# --------------------------------------------------------------------------- #
# 方程识别：支持 ∇² / Δ 记号的热传导 / 波动 / Laplace
# --------------------------------------------------------------------------- #
_RE_LAPLACIAN = re.compile(r"\\nabla\s*\^?\s*\{?\s*2|\\Delta(?![a-zA-Z])")
_RE_BARE_NABLA = re.compile(r"\\nabla")
_RE_WAVE_ORDER = re.compile(r"\\partial\s*\^\s*\{?\s*2|u_\{?\s*t\s*t|u''")
_RE_HEAT_ORDER = re.compile(
    r"\\frac\s*\{\s*\\partial\s*\w+\s*\}\s*\{\s*\\partial\s*t\s*\}"
    r"|u_\{?\s*t\s*\}?(?![a-zA-Z])"
)


def _split_equation(s):
    parts = s.split("=")
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError(
            r"方程需要恰好一个等号，例如 \frac{\partial u}{\partial t} = \alpha \nabla^2 u"
        )
    return parts[0].strip(), parts[1].strip()


def _has_laplacian(s):
    return bool(_RE_LAPLACIAN.search(s))


def _parse_pde_equation(eq):
    """从 LaTeX 方程识别类型与系数（支持 \nabla^2、\Delta、u_t、u_{tt} 记号）。

    返回 (ptype, coeff, info)：coeff 为数值时直接采用（heat=α，wave=c）；
    为 None 时（方程里写的是 \alpha、c 等符号）回退到系数输入框。

    说明：sympy 的 parse_latex 会把 \nabla 当普通符号、把 \partial^2 的二阶
    导数解析成乱码，故这里基于 LaTeX 原文做正则识别，仅对系数部分做
    sympy 解析。
    """
    s = re.sub(r"\\left|\\right", "", eq.strip())
    if not _has_laplacian(s):
        if _RE_BARE_NABLA.search(s):
            raise ValueError(
                r"检测到梯度算子 \nabla：请用二阶算子 \nabla^2（或 \Delta）表示 Laplacian"
            )
        raise ValueError(r"方程中未识别到 Laplacian（\nabla^2 或 \Delta）")

    is_wave = bool(_RE_WAVE_ORDER.search(s))
    is_heat = (not is_wave) and bool(_RE_HEAT_ORDER.search(s))

    lhs, rhs = _split_equation(s)
    if _has_laplacian(lhs) and _has_laplacian(rhs):
        raise ValueError("方程两侧都出现了 Laplacian，无法识别")
    lap_side = lhs if _has_laplacian(lhs) else rhs
    other = rhs if lap_side is lhs else lhs

    # 提取系数：去掉 Laplacian 算子与依变量 u 后解析剩余部分
    coeff_raw = _RE_LAPLACIAN.sub("", lap_side)
    coeff_raw = re.sub(r"(?<![a-zA-Z\\])u(?![a-zA-Z])", "", coeff_raw).strip() or "1"
    expr, syms = latex_to_python(coeff_raw)
    coeff = float(expr) if not syms else None

    if is_wave:
        c = None if coeff is None else coeff ** 0.5
        info = "已从方程识别：波动方程" + (
            f"，c = {c:.4g}" if c is not None else "，c 取自下方输入框")
        return "wave", c, info
    if is_heat:
        info = "已从方程识别：热传导方程" + (
            f"，α = {coeff:.4g}" if coeff is not None else "，α 取自下方输入框")
        return "heat", coeff, info

    # 无时间导数 -> Laplace；另一侧必须为 0（泊松方程暂不支持）
    other_expr, other_syms = latex_to_python(other or "0")
    if other_syms or float(other_expr) != 0.0:
        raise ValueError("暂不支持泊松方程（Δu = 非零右端项），右端请填 0")
    return "laplace", None, "已从方程识别：Laplace 方程"


def _pde_boundary_mask(shape):
    """返回标记边界面（含棱、角）的布尔数组。"""
    mask = np.zeros(shape, dtype=bool)
    for a in range(len(shape)):
        for face in (0, -1):
            sl = [slice(None)] * len(shape)
            sl[a] = face
            mask[tuple(sl)] = True
    return mask


def _pde_plot(snaps, labels, dims, L, var_names, ptype):
    """把快照序列画成一张图：1D 折线，2D/3D 热图子图网格。"""
    m = len(snaps)
    cmap = plt.get_cmap("viridis")
    if dims == 1:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.linspace(0.0, L, snaps.shape[1])
        for i in range(m):
            ax.plot(x, snaps[i], color=cmap(i / max(m - 1, 1)), lw=1.8,
                    label=labels[i])
        ax.set_xlabel(var_names[0])
        ax.set_ylabel("u")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ncol = min(3, m)
        nrow = (m + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                                 squeeze=False)
        for i in range(m):
            ax = axes[i // ncol][i % ncol]
            field = snaps[i] if dims == 2 else snaps[i][snaps.shape[1] // 2]
            im = ax.imshow(field.T, origin="lower", extent=[0, L, 0, L],
                           cmap="viridis", aspect="auto")
            ax.set_title(labels[i], fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
        for j in range(m, nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle(f"{'z = L/2 截面 · ' if dims == 3 else ''}"
                     f"{var_names[-1]} 方向中心切片" if dims == 3 else "", fontsize=9)
    fig.tight_layout()
    return fig_to_base64(fig)


def api_pde(payload: dict) -> dict:
    """PDE 求解：类型/维数/LaTeX 初值与边界 -> n 维求解器 -> 快照图。"""
    try:
        dims = int(payload.get("dims", 2))
    except (TypeError, ValueError):
        return {"ok": False, "error": "维数必须是整数"}
    if dims not in (1, 2, 3):
        return {"ok": False, "error": "空间维数仅支持 1 / 2 / 3"}

    # 方程输入框（可选）：识别 ∇²/Δ 记号，自动覆盖类型与系数
    eq_raw = (payload.get("eq") or "").strip()
    eq_coeff = None
    eq_info = None
    if eq_raw:
        try:
            ptype, eq_coeff, eq_info = _parse_pde_equation(eq_raw)
        except ValueError as e:
            return {"ok": False, "error": f"方程识别失败：{e}"}
    else:
        ptype = payload.get("type")
        if ptype not in ("heat", "wave", "laplace"):
            return {"ok": False, "error": "未知方程类型（heat / wave / laplace）"}
    if ptype == "laplace" and dims < 2:
        return {"ok": False, "error": "Laplace 方程至少需要 2 个空间维数"}

    var_names = PDE_VAR_NAMES[dims]
    try:
        L = _pde_num(payload, "L", 1.0, lo=1e-9, what="区域边长 L")
        n = int(_pde_num(payload, "n", 61, lo=5, what="每维网格点数"))
        cap_n = PDE_GRID_CAPS[dims]
        if n > cap_n:
            return {"ok": False,
                    "error": f"{dims} 维网格每维最多 {cap_n} 点（内存限制），当前 {n}"}
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    grid_desc = {1: f"{n} 点", 2: f"{n}×{n}", 3: f"{n}×{n}×{n}"}[dims]

    axes = [np.linspace(0.0, L, n)] * dims
    grids = np.meshgrid(*axes, indexing="ij")
    dx = L / (n - 1)

    try:
        u0 = _latex_field(payload.get("latex", "0"), var_names, grids, "初值/初始猜测")
        bnd = _latex_field(payload.get("bc", "0") or "0", var_names, grids, "边界条件")
    except Exception as e:
        return {"ok": False, "error": f"公式解析失败：{e}"}

    try:
        if ptype == "heat":
            if eq_raw and eq_coeff is not None:
                alpha = eq_coeff
            else:
                alpha = _pde_num(payload, "coeff", 1.0, lo=1e-12, what="α 热扩散系数")
            dt = _pde_num(payload, "dt", 0.25 * dx * dx / dims, lo=1e-12)
            steps = int(_pde_num(payload, "steps", 500, lo=0))
            if steps > PDE_STEP_CAPS[dims]:
                return {"ok": False,
                        "error": f"{dims} 维演化步数最多 {PDE_STEP_CAPS[dims]}，当前 {steps}"}
            times, snaps = solve_heat_nd(u0, alpha, dx, dt, steps,
                                         boundary=bnd, n_snapshots=6)
            r_sum = alpha * dt * dims / dx**2
            labels = [f"t = {t:.4g}" for t in times]
            meta = (f"{dims} 维热传导 · 网格 {grid_desc}"
                    + f" · dx={dx:.4g} · dt={dt:.4g} · {steps} 步 · "
                      f"Σ r_i = {r_sum:.3g}（≤ 0.5）")
            scheme = "FTCS：u^{n+1} = u^n + dt·α·Δu^n"
            python_src = "u0 = " + sp.printing.lambdarepr.lambdarepr(
                _latex_expr(payload.get("latex", "0"), var_names))
        elif ptype == "wave":
            if eq_raw and eq_coeff is not None:
                c = eq_coeff
            else:
                c = _pde_num(payload, "coeff", 1.0, lo=1e-12, what="c 波速")
            dt = _pde_num(payload, "dt", 0.8 / (c * np.sqrt(dims) / dx), lo=1e-12)
            steps = int(_pde_num(payload, "steps", 300, lo=0))
            if steps > PDE_STEP_CAPS[dims]:
                return {"ok": False,
                        "error": f"{dims} 维演化步数最多 {PDE_STEP_CAPS[dims]}，当前 {steps}"}
            times, snaps = solve_wave_nd(u0, 0.0, c, dx, dt, steps,
                                         boundary=bnd, n_snapshots=6)
            cfl = c * dt * np.sqrt(dims) / dx
            labels = [f"t = {t:.4g}" for t in times]
            meta = (f"{dims} 维波动 · 网格 {grid_desc}"
                    + f" · dx={dx:.4g} · dt={dt:.4g} · {steps} 步 · "
                      f"CFL = {cfl:.3g}（≤ 1）")
            scheme = "蛙跳：u^{n+1} = 2u^n − u^{n−1} + (c·dt)²·Δu^n"
            python_src = "u0 = " + sp.printing.lambdarepr.lambdarepr(
                _latex_expr(payload.get("latex", "0"), var_names))
        else:  # laplace
            tol = _pde_num(payload, "tol", 1e-6, lo=1e-12)
            max_iter = int(_pde_num(payload, "maxiter", 20000, lo=1, hi=200000))
            mask = _pde_boundary_mask(u0.shape)
            u_guess = np.where(mask, bnd, u0)
            snaps, iters = solve_laplace_nd(u_guess, dx=dx, tol=tol,
                                            max_iter=max_iter, n_snapshots=6)
            labels = [f"iter = {k}" for k in iters]
            meta = (f"{dims} 维 Laplace · 网格 {grid_desc}"
                    + f" · tol={tol:.3g} · 收敛 {int(iters[-1])} 轮")
            scheme = "红黑 Gauss–Seidel：u ← 邻居加权平均（SOR ω 可调）"
            python_src = "u0 = " + sp.printing.lambdarepr.lambdarepr(
                _latex_expr(payload.get("latex", "0"), var_names))
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"求解失败：{e}"}

    try:
        plot = _pde_plot(np.asarray(snaps), labels, dims, L, var_names, ptype)
    except Exception as e:
        return {"ok": False, "error": f"绘图失败：{e}"}

    if eq_info:
        meta = eq_info + " · " + meta

    return {
        "ok": True,
        "mode": "pde",
        "pde_type": ptype,
        "dims": dims,
        "var_names": var_names,
        "python_src": python_src,
        "scheme": scheme,
        "meta": meta,
        "n_snapshots": int(np.asarray(snaps).shape[0]),
        "plot": plot,
    }


def _latex_expr(latex, var_names):
    """仅供 python_src 展示：LaTeX -> sympy 表达式（e 视为 Euler 数）。"""
    expr, _ = latex_to_python(latex or "0")
    if "e" not in var_names:
        expr = expr.subs(sp.Symbol("e"), sp.E)
    return expr


# --------------------------------------------------------------------------- #
# HTTP 服务
# --------------------------------------------------------------------------- #
WEB_DIR = os.path.dirname(os.path.abspath(__file__))


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path, content_type):
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            self.send_error(404, "Not Found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        if route in ("/", "/index.html"):
            self._send_file(os.path.join(WEB_DIR, "templates", "index.html"),
                            "text/html; charset=utf-8")
        elif route.startswith("/static/"):
            rel = route[len("/static/"):]
            # 防目录穿越
            rel = rel.replace("\\", "/")
            full = os.path.normpath(os.path.join(WEB_DIR, "static", rel))
            base = os.path.normpath(os.path.join(WEB_DIR, "static"))
            if not full.startswith(base):
                self.send_error(403, "Forbidden")
                return
            ctype = "text/css; charset=utf-8" if rel.endswith(".css") else "application/javascript; charset=utf-8"
            self._send_file(full, ctype)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in ("/api/solve", "/api/eval", "/api/pde"):
            self.send_error(404, "Not Found")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self._send_json({"ok": False, "error": f"请求体解析失败：{e}"}, status=400)
            return

        if parsed.path == "/api/solve":
            self._send_json(api_solve(payload))
        elif parsed.path == "/api/eval":
            self._send_json(api_eval(payload))
        else:
            self._send_json(api_pde(payload))

    def log_message(self, fmt, *args):  # 静默默认日志
        pass


def main():
    parser = argparse.ArgumentParser(description="diffeq-toolkit 网页公式求解服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # matplotlib 配置目录放到 web 目录下，避免无头沙箱写入 HOME 被拒
    mpl_dir = os.path.join(WEB_DIR, ".mplconfig")
    os.makedirs(mpl_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_dir

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"diffeq-toolkit web 已启动：{url}")
    print("在网页中输入 LaTeX 公式即可求解/计算。按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()

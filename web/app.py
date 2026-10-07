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
        if parsed.path not in ("/api/solve", "/api/eval"):
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
        else:
            self._send_json(api_eval(payload))

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

"""示例 1：指数衰减方程 dy/dt = -k·y 的三种定步长方法对比。

运行：python examples/ode_decay.py
输出：docs/images/ode_decay.png 与各方法误差对比
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffeq.plotting import setup_cjk_font
from diffeq.ode import solve_ode
from diffeq.diagnostics import endpoint_error

setup_cjk_font()  # 启用中文字体（若系统可用）

K = 0.5
F = lambda t, y: -K * y            # noqa: E731
EXACT = lambda t: 2.0 * np.exp(-K * t)  # noqa: E731


def main():
    out_dir = Path(__file__).resolve().parents[1] / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("粗步长（h = 0.4，仅 10 步）下的直观对比：")
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    t_fine = np.linspace(0, 4, 200)
    ax.plot(t_fine, EXACT(t_fine), "k--", lw=1.5, label="精确解")

    for method, marker in [("euler", "o"), ("heun", "s"), ("rk4", "^")]:
        t, y = solve_ode(F, 2.0, (0.0, 4.0), n_steps=10, method=method)
        ax.plot(t, y, marker=marker, ms=4, lw=1.2, label=f"{method}（h=0.4）")

    print("细步长（n = 400 步）下的端点误差：")
    for method in ("euler", "heun", "rk4"):
        err = endpoint_error(F, 2.0, (0.0, 4.0), method, 400, EXACT)
        print(f"  {method:>6s}: {err:.3e}")

    ax.set_xlabel("$t$")
    ax.set_ylabel("$y$")
    ax.set_title(r"$dy/dt = -0.5y,\ y(0)=2$：三种定步长方法对比")
    ax.grid(alpha=0.3)
    ax.legend()
    out = out_dir / "ode_decay.png"
    fig.savefig(out, dpi=150)
    print(f"图片已保存: {out}")


if __name__ == "__main__":
    main()

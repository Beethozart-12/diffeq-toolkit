"""示例 2：谐振子 x'' + x = 0 的 RK4 求解、时间序列与相图。

二阶方程化为方程组：s = [x, v]，s' = [v, -x]。
运行：python examples/ode_harmonic.py
输出：docs/images/ode_harmonic.png 与能量漂移检查
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

setup_cjk_font()  # 启用中文字体（若系统可用）


def f(t, s):
    x, v = s
    return np.array([v, -x])


def main():
    out_dir = Path(__file__).resolve().parents[1] / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    t, y = solve_ode(f, [1.0, 0.0], (0.0, 4 * np.pi), n_steps=400)
    energy = y[:, 0] ** 2 + y[:, 1] ** 2
    print(f"RK4 求解谐振子：终点 x(2π) = {y[-1, 0]:.10f}（精确值 1）")
    print(f"能量守恒检查：max|E - 1| = {np.max(np.abs(energy - 1.0)):.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    axes[0].plot(t, y[:, 0], label="$x(t)$ 位移")
    axes[0].plot(t, y[:, 1], label="$v(t)$ 速度")
    axes[0].set_xlabel("$t$")
    axes[0].set_title("时间序列")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(y[:, 0], y[:, 1], lw=1.2, color="crimson")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$v$")
    axes[1].set_title("相图（能量应守恒）")
    axes[1].grid(alpha=0.3)
    axes[1].set_aspect("equal")

    out = out_dir / "ode_harmonic.png"
    fig.savefig(out, dpi=150)
    print(f"图片已保存: {out}")


if __name__ == "__main__":
    main()

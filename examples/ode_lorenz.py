"""示例 3：Lorenz 系统的 RK45 自适应求解与三维相轨迹。

Lorenz 系统：
    x' = σ(y - x)
    y' = x(ρ - z) - y
    z' = xy - βz

运行：python examples/ode_lorenz.py
输出：docs/images/ode_lorenz.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffeq.plotting import setup_cjk_font
from diffeq.ode import solve_ivp

setup_cjk_font()  # 启用中文字体（若系统可用）

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0


def lorenz(t, s):
    x, y, z = s
    return np.array([
        SIGMA * (y - x),
        x * (RHO - z) - y,
        x * y - BETA * z,
    ])


def main():
    out_dir = Path(__file__).resolve().parents[1] / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    t, y = solve_ivp(lorenz, [1.0, 1.0, 1.0], (0.0, 25.0), rtol=1e-8, atol=1e-10)
    print(f"自适应 RK45 在 t ∈ [0, 25] 内共接受 {len(t) - 1} 步")
    print(f"终态: x={y[-1, 0]:.3f}, y={y[-1, 1]:.3f}, z={y[-1, 2]:.3f}")

    fig = plt.figure(figsize=(7.5, 6.2))
    ax = fig.add_subplot(projection="3d")
    ax.plot(y[:, 0], y[:, 1], y[:, 2], lw=0.5, color="crimson")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_title("Lorenz 吸引子（RK45 自适应步长）")

    out = out_dir / "ode_lorenz.png"
    fig.savefig(out, dpi=150)
    print(f"图片已保存: {out}")


if __name__ == "__main__":
    main()

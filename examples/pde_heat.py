"""示例 4：一维热传导方程 —— FTCS 显式与 Crank–Nicolson 隐式对比。

    u_t = α·u_xx,  x ∈ (0, 1),  u(0,t) = u(1,t) = 0,  u(x,0) = sin(πx)
    精确解：u(x,t) = sin(πx)·exp(-π²t)

运行：python examples/pde_heat.py
输出：docs/images/pde_heat.png 与两种格式的最大误差
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffeq.plotting import setup_cjk_font
from diffeq.pde import solve_heat_ftcs, solve_heat_crank_nicolson

setup_cjk_font()  # 启用中文字体（若系统可用）

ALPHA = 1.0
T = 0.02


def exact(x, t):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)


def main():
    out_dir = Path(__file__).resolve().parents[1] / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    nx = 100
    x = np.linspace(0.0, 1.0, nx + 1)
    dx = 1.0 / nx
    u0 = np.sin(np.pi * x)

    # FTCS：显式，受稳定性限制 r ≤ 0.5
    dt_ftcs = 0.45 * dx**2
    n_ftcs = int(round(T / dt_ftcs))
    u_ftcs = solve_heat_ftcs(u0, ALPHA, dx, dt_ftcs, n_ftcs)
    T_ftcs = n_ftcs * dt_ftcs

    # Crank–Nicolson：隐式，无条件稳定，可用大得多的时间步
    dt_cn = 1.0e-3
    n_cn = int(round(T / dt_cn))
    u_cn = solve_heat_crank_nicolson(u0, ALPHA, dx, dt_cn, n_cn)

    err_ftcs = np.max(np.abs(u_ftcs[-1] - exact(x, T_ftcs)))
    err_cn = np.max(np.abs(u_cn[-1] - exact(x, T)))
    print(f"FTCS          (dt = {dt_ftcs:.2e}, {n_ftcs} 步): 最大误差 = {err_ftcs:.3e}")
    print(f"Crank-Nicolson (dt = {dt_cn:.2e}, {n_cn} 步): 最大误差 = {err_cn:.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)

    t_grid = np.linspace(0.0, T, u_cn.shape[0])
    mesh = axes[0].pcolormesh(x, t_grid, u_cn, shading="auto", cmap="inferno")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$t$")
    axes[0].set_title("Crank–Nicolson 数值解 $u(x,t)$")
    fig.colorbar(mesh, ax=axes[0], label="$u$")

    axes[1].plot(x, exact(x, T), "k--", lw=1.5, label="精确解")
    axes[1].plot(x, u_ftcs[-1], "o", ms=3, label=f"FTCS（t = {T_ftcs:.4f}）")
    axes[1].plot(x, u_cn[-1], "s", ms=3, label="Crank–Nicolson")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$u$")
    axes[1].set_title(f"t = {T} 时刻温度分布")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    out = out_dir / "pde_heat.png"
    fig.savefig(out, dpi=150)
    print(f"图片已保存: {out}")


if __name__ == "__main__":
    main()

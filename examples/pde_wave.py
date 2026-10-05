"""示例 5：两端固定的弦振动（一维波动方程）的显式差分求解。

    u_tt = c²·u_xx,  x ∈ (0, 1),  u(0,t) = u(1,t) = 0
    初值：u(x,0) = sin(πx),  u_t(x,0) = 0
    精确解：u(x,t) = sin(πx)·cos(πct)（驻波）

运行：python examples/pde_wave.py
输出：docs/images/pde_wave.png 与 t = 2 时刻的最大误差
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from diffeq.pde import solve_wave
from diffeq.plotting import plot_snapshots

C = 1.0
L = 1.0


def main():
    out_dir = Path(__file__).resolve().parents[1] / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    nx = 200
    x = np.linspace(0.0, L, nx + 1)
    dx = L / nx
    dt = 0.004                # CFL = c·dt/dx = 0.8 < 1
    n_steps = 500             # T = 2.0，驻波恰好完成一个整周期回到初始位置

    u0 = np.sin(np.pi * x)
    v0 = np.zeros_like(x)
    u = solve_wave(u0, v0, C, dx, dt, n_steps)

    T = n_steps * dt
    exact = np.sin(np.pi * x) * np.cos(np.pi * C * T)
    err = np.max(np.abs(u[-1] - exact))
    print(f"CFL = {C * dt / dx:.2f}，T = {T}（一个完整周期）")
    print(f"t = {T} 时刻与精确解的最大误差 = {err:.3e}")

    t_grid = np.arange(n_steps + 1) * dt
    fig, ax = plot_snapshots(
        u, x, t_grid, n=6,
        title=f"两端固定弦的振动（CFL = {C * dt / dx:.2f}）",
    )
    ax.set_ylim(-1.1, 1.1)
    out = out_dir / "pde_wave.png"
    fig.savefig(out, dpi=150)
    print(f"图片已保存: {out}")


if __name__ == "__main__":
    main()

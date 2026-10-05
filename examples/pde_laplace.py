"""示例 6：矩形域上二维 Laplace 方程的迭代求解。

    u_xx + u_yy = 0,  (x, y) ∈ (0, 1)²
    边界条件：u(x,0) = 0, u(x,1) = x, u(0,y) = 0, u(1,y) = y
    精确解：u = x·y

运行：python examples/pde_laplace.py
输出：docs/images/pde_laplace.png 与收敛信息
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from diffeq.pde import solve_laplace
from diffeq.plotting import plot_laplace

N = 40


def main():
    out_dir = Path(__file__).resolve().parents[1] / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0.0, 1.0, N + 1)
    X, Y = np.meshgrid(x, x)

    # 边界条件：u = x·y；内部以 0 为迭代初值
    guess = np.zeros((N + 1, N + 1))
    guess[0, :] = 0.0         # u(x, 0) = 0
    guess[-1, :] = x          # u(x, 1) = x
    guess[:, 0] = 0.0         # u(0, y) = 0
    guess[:, -1] = x          # u(1, y) = y

    u, n_iter = solve_laplace(guess, tol=1e-11, max_iter=20000)
    err = np.max(np.abs(u - X * Y))
    print(f"Gauss–Seidel 迭代 {n_iter} 次后收敛")
    print(f"与精确解 u = x·y 的最大误差 = {err:.3e}")

    fig, ax = plot_laplace(
        u, x, x,
        title=f"Laplace 方程数值解（{N}×{N} 网格，迭代 {n_iter} 次）",
        save=str(out_dir / "pde_laplace.png"),
    )
    print(f"图片已保存: {out_dir / 'pde_laplace.png'}")


if __name__ == "__main__":
    main()

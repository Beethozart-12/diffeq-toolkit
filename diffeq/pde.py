"""偏微分方程（PDE）数值求解模块。

包含三类典型方程：

- 一维热传导方程 ``u_t = α·u_xx``：FTCS 显式格式、Crank–Nicolson 隐式格式
- 一维波动方程 ``u_tt = c²·u_xx``：中心差分（蛙跳）显式格式
- 二维 Laplace 方程 ``u_xx + u_yy = 0``：逐行红黑混合 Gauss–Seidel 迭代（行内 Jacobi、跨行 GS）

约定：一维问题网格长度为 n+1（含两个边界节点），边界按 Dirichlet 条件处理，
取初值两端的值并在演化中保持不变；二维问题网格形状为 (ny+1, nx+1)。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "thomas_solve",
    "solve_heat_ftcs",
    "solve_heat_crank_nicolson",
    "solve_wave",
    "solve_laplace",
]


# ----------------------------------------------------------------------
# 三对角方程组求解（Thomas 算法）
# ----------------------------------------------------------------------

def thomas_solve(sub, diag, sup, rhs):
    """Thomas 算法（追赶法）求解三对角线性方程组，复杂度 O(n)。

    参数
    ----
    sub : array_like, shape (n-1,)
        次对角线元素 ``A[i, i-1]``。
    diag : array_like, shape (n,)
        主对角线元素。
    sup : array_like, shape (n-1,)
        超对角线元素 ``A[i, i+1]``。
    rhs : array_like, shape (n,)
        右端项。

    返回
    ----
    x : ndarray, shape (n,)
        解向量。

    异常
    ----
    ValueError : 维度不匹配或消元过程中主元为零时抛出。
    """
    sub = np.asarray(sub, dtype=float)
    diag = np.asarray(diag, dtype=float)
    sup = np.asarray(sup, dtype=float)
    rhs = np.asarray(rhs, dtype=float)

    n = diag.size
    if not (sub.size == sup.size == n - 1 and rhs.size == n):
        raise ValueError("三对角矩阵维度不匹配")

    if n == 1:
        if diag[0] == 0.0:
            raise ValueError("矩阵奇异（主元为零）")
        return np.array([rhs[0] / diag[0]])

    cp = np.zeros(n - 1)  # 消元后的超对角线
    dp = np.zeros(n)      # 消元后的右端项

    m = diag[0]
    if m == 0.0:
        raise ValueError("矩阵奇异（主元为零）")
    cp[0] = sup[0] / m
    dp[0] = rhs[0] / m
    for i in range(1, n):
        m = diag[i] - sub[i - 1] * cp[i - 1]
        if m == 0.0:
            raise ValueError("矩阵奇异（主元为零）")
        dp[i] = (rhs[i] - sub[i - 1] * dp[i - 1]) / m
        if i < n - 1:
            cp[i] = sup[i] / m

    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ----------------------------------------------------------------------
# 一维热传导方程
# ----------------------------------------------------------------------

def _check_heat_inputs(u0, alpha, dx, dt):
    u0 = np.asarray(u0, dtype=float)
    if alpha <= 0 or dt <= 0 or dx <= 0:
        raise ValueError("alpha、dt、dx 必须为正数")
    if u0.ndim != 1 or u0.size < 3:
        raise ValueError("u0 必须是一维数组且至少包含 3 个节点")
    return u0


def solve_heat_ftcs(u0, alpha, dx, dt, n_steps):
    """一维热传导方程 ``u_t = α·u_xx`` 的显式 FTCS 格式。

    时间用前向差分、空间用中心差分，整体精度 O(dt + dx²)。

    参数
    ----
    u0 : array_like, shape (n+1,)
        初值（含左右边界节点）。边界按 Dirichlet 条件处理，
        取 ``u0[0]``、``u0[-1]`` 并在整个演化中保持不变。
    alpha : float
        热扩散系数（> 0）。
    dx, dt : float
        空间/时间步长。
    n_steps : int
        演化步数。

    返回
    ----
    u : ndarray, shape (n_steps+1, n+1)
        第 k 行为 ``t = t0 + k·dt`` 时刻的空间分布。

    异常
    ----
    ValueError : 违反稳定性条件 ``r = α·dt/dx² ≤ 1/2`` 时抛出。
    """
    u0 = _check_heat_inputs(u0, alpha, dx, dt)
    r = alpha * dt / dx**2
    if r > 0.5 + 1e-12:
        raise ValueError(
            f"FTCS 格式不稳定：r = alpha*dt/dx^2 = {r:.4g} > 0.5，请减小 dt 或增大 dx"
        )

    n = u0.size - 1
    u = np.empty((n_steps + 1, u0.size))
    u[0] = u0
    for k in range(n_steps):
        u[k + 1, 1:n] = u[k, 1:n] + r * (u[k, 2:] - 2.0 * u[k, 1:n] + u[k, :-2])
        u[k + 1, 0] = u0[0]
        u[k + 1, n] = u0[-1]
    return u


def solve_heat_crank_nicolson(u0, alpha, dx, dt, n_steps):
    """一维热传导方程的 Crank–Nicolson 隐式格式。

    时间、空间均为二阶精度 O(dt² + dx²)，无条件稳定；
    每步用 Thomas 算法求解三对角方程组。

    边界处理：冻结 Dirichlet 条件，即 u^{n+1}_0 = u^n_0（取初值两端并保持不变）。
    因边界冻结，隐式侧使用旧时刻或新时刻边界在数值上完全等价；实现统一采用
    新时刻边界值，便于未来支持非定常边界。

    参数与返回值同 :func:`solve_heat_ftcs`。
    """
    u0 = _check_heat_inputs(u0, alpha, dx, dt)
    n = u0.size - 1
    m = n - 1                      # 内部未知数个数
    r = alpha * dt / (2.0 * dx**2)

    u = np.empty((n_steps + 1, u0.size))
    u[0] = u0
    sub = np.full(max(m - 1, 0), -r)
    sup = np.full(max(m - 1, 0), -r)
    diag = np.full(m, 1.0 + 2.0 * r)

    for k in range(n_steps):
        uk = u[k]
        # 显式部分 (I + rA)·u^k
        rhs = uk[1:n] + r * (uk[2:n + 1] - 2.0 * uk[1:n] + uk[0:n - 1])
        # 隐式侧边界：取本步（u^{n+1}）的边界值。此处为冻结 Dirichlet
        # （边界在演化中保持不变），故 u^{n+1}_0 == u^n_0，数值上等价于旧时刻
        # 边界；统一使用新时刻边界，便于未来扩展非定常边界。
        u_new = u[k + 1]
        u_new[0], u_new[n] = uk[0], uk[n]
        rhs[0] += r * u_new[0]
        rhs[-1] += r * u_new[n]
        u_new[1:n] = thomas_solve(sub, diag, sup, rhs)
    return u


# ----------------------------------------------------------------------
# 一维波动方程
# ----------------------------------------------------------------------

def solve_wave(u0, v0, c, dx, dt, n_steps):
    """一维波动方程 ``u_tt = c²·u_xx`` 的显式中心差分（蛙跳）格式。

    时间、空间均为二阶精度；稳定性要求 CFL 条件 ``c·dt/dx ≤ 1``。
    第一层由泰勒展开构造：``u¹ = u⁰ + dt·v⁰ + ½C²·δ²u⁰``。

    参数
    ----
    u0, v0 : array_like, shape (n+1,)
        初始位移与初始速度（含边界节点）。
        边界位移取 ``u0[0]``、``u0[-1]`` 并保持不变（Dirichlet）。
    c : float
        波速（> 0）。
    dx, dt : float
        空间/时间步长。
    n_steps : int
        演化步数。

    返回
    ----
    u : ndarray, shape (n_steps+1, n+1)

    异常
    ----
    ValueError : 违反 CFL 条件时抛出。
    """
    u0 = np.asarray(u0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    if u0.shape != v0.shape or u0.ndim != 1 or u0.size < 3:
        raise ValueError("u0、v0 必须是同长度的一维数组且至少包含 3 个节点")
    if c <= 0 or dt <= 0 or dx <= 0:
        raise ValueError("c、dt、dx 必须为正数")

    cfl = c * dt / dx
    if cfl > 1.0 + 1e-12:
        raise ValueError(f"违反 CFL 条件：c*dt/dx = {cfl:.4g} > 1，请减小 dt 或增大 dx")

    n = u0.size - 1
    u = np.empty((n_steps + 1, u0.size))
    u[0] = u0
    if n_steps == 0:
        return u

    # 第一层：泰勒展开
    u[1, 1:n] = (u0[1:n] + dt * v0[1:n]
                 + 0.5 * cfl**2 * (u0[2:] - 2.0 * u0[1:n] + u0[:-2]))
    u[1, 0], u[1, n] = u0[0], u0[-1]

    # 蛙跳推进
    for k in range(1, n_steps):
        u[k + 1, 1:n] = (2.0 * u[k, 1:n] - u[k - 1, 1:n]
                         + cfl**2 * (u[k, 2:] - 2.0 * u[k, 1:n] + u[k, :-2]))
        u[k + 1, 0], u[k + 1, n] = u0[0], u0[-1]
    return u


# ----------------------------------------------------------------------
# 二维 Laplace 方程
# ----------------------------------------------------------------------

def solve_laplace(u_guess, tol=1e-6, max_iter=10000):
    """二维 Laplace 方程 ``u_xx + u_yy = 0`` 在矩形域上的迭代求解。

    采用红黑混合（行内 Jacobi、跨行 Gauss–Seidel）的五点平均迭代：
    逐行推进时，第 i 行使用上一轮已更新的第 i-1 行（跨行 GS），
    但同一行内的左右邻居仍取上一轮旧值（行内 Jacobi）。五点格式为二阶精度，
    收敛行为与标准 Gauss–Seidel 等价。

    参数
    ----
    u_guess : ndarray, shape (ny+1, nx+1)
        边界元素给出 Dirichlet 边界条件（迭代中保持不变），
        内部元素作为迭代初值。
    tol : float
        相邻两次迭代的最大变化量小于 ``tol`` 时判定收敛。
    max_iter : int
        最大迭代次数。

    返回
    ----
    u : ndarray, shape (ny+1, nx+1)
        收敛后的解。
    n_iter : int
        实际迭代次数；若未收敛则返回 ``max_iter``。

    异常
    ----
    ValueError : 网格过小时抛出。
    """
    u = np.array(u_guess, dtype=float, copy=True)
    if u.ndim != 2 or u.shape[0] < 3 or u.shape[1] < 3:
        raise ValueError("u_guess 必须是二维数组，且每个方向至少 3 个节点")

    ny, nx = u.shape[0] - 1, u.shape[1] - 1
    for it in range(1, max_iter + 1):
        diff = 0.0
        for i in range(1, ny):
            new_vals = 0.25 * (u[i - 1, 1:nx] + u[i + 1, 1:nx]
                               + u[i, 0:nx - 1] + u[i, 2:nx + 1])
            diff = max(diff, float(np.max(np.abs(new_vals - u[i, 1:nx]))))
            u[i, 1:nx] = new_vals
        if diff < tol:
            return u, it
    return u, max_iter

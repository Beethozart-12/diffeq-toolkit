"""偏微分方程（PDE）数值求解模块。

包含三类典型方程：

- 热传导方程 ``u_t = α·Δu``：一维 FTCS 显式格式、一维 Crank–Nicolson 隐式格式、
  **任意空间维数**的 FTCS 显式格式（:func:`solve_heat_nd`）
- 波动方程 ``u_tt = c²·Δu``：一维中心差分（蛙跳）显式格式、
  **任意空间维数**的蛙跳格式（:func:`solve_wave_nd`）
- Laplace 方程 ``Δu = 0``：二维逐点（自然顺序）Gauss–Seidel 迭代、
  **任意空间维数**的红黑 Gauss–Seidel/SOR 迭代（:func:`solve_laplace_nd`）

约定
----
- 一维问题网格长度为 n+1（含两个边界节点），边界按 Dirichlet 条件处理，
  取初值两端的值并在演化中保持不变；二维问题网格形状为 (ny+1, nx+1)。
- ``*_nd`` 系列支持任意空间维数 k ≥ 1：网格 ``u0`` 形状为 ``(n1, ..., nk)``
  （每个维度都含边界节点）；空间步长 ``dx`` 可为标量或长度 k 的序列。
- ``*_nd`` 系列的边界条件接受标量或与 ``u0`` 同形状的数组（Dirichlet，
  演化中保持不变），返回按时间/迭代均匀抽样的快照序列。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "thomas_solve",
    "solve_heat_ftcs",
    "solve_heat_crank_nicolson",
    "solve_wave",
    "solve_laplace",
    "solve_heat_nd",
    "solve_wave_nd",
    "solve_laplace_nd",
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

    采用逐点（自然顺序 lexicographic）Gauss–Seidel 迭代：按 i=1..ny-1、j=1..nx-1 的
    顺序逐个更新内部节点，更新后立即用新值参与后续邻居的计算（即 u[i, j-1] 与
    u[i-1, j] 均为本轮已更新的新值）。这是标准 Gauss–Seidel 的严格实现，收敛速度通常
    快于 Jacobi；代价是内层为逐点 Python 循环、较向量化 Jacobi 慢，属「教学清晰性优先」
    的取舍。五点格式为二阶精度。

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
        # 逐点（自然顺序）Gauss–Seidel：u[i, j-1] 与 u[i-1, j] 已被本轮更新为新值
        for i in range(1, ny):
            for j in range(1, nx):
                new_val = 0.25 * (u[i - 1, j] + u[i + 1, j]
                                   + u[i, j - 1] + u[i, j + 1])
                delta = abs(new_val - u[i, j])
                if delta > diff:
                    diff = delta
                u[i, j] = new_val
        if diff < tol:
            return u, it
    return u, max_iter


# ----------------------------------------------------------------------
# n 维热传导 / 波动 / Laplace（任意空间维数，向量化实现）
# ----------------------------------------------------------------------

def _as_float_grid(u0, name="u0"):
    """校验并转换网格数组：每个方向至少 3 个节点（含两个边界节点）。"""
    u = np.asarray(u0, dtype=float)
    if u.ndim < 1 or min(u.shape) < 3:
        raise ValueError(f"{name} 每个方向至少需要 3 个节点（含边界），实际形状 {u.shape}")
    return u


def _dx_sequence(dx, k):
    """把标量或长度 k 的序列统一成长度 k 的正数数组。"""
    dxs = np.atleast_1d(np.asarray(dx, dtype=float))
    if dxs.size == 1:
        dxs = np.full(k, float(dxs[0]))
    elif dxs.size != k:
        raise ValueError(f"dx 应为标量或长度 {k} 的序列，实际长度 {dxs.size}")
    if np.any(dxs <= 0):
        raise ValueError("dx 必须为正数")
    return dxs


def _resolve_boundary(boundary, u):
    """把标量或同形状数组统一成与 u 同形状的边界值数组。"""
    if np.isscalar(boundary):
        return np.full(u.shape, float(boundary))
    b = np.asarray(boundary, dtype=float)
    if b.shape != u.shape:
        raise ValueError(f"boundary 形状 {b.shape} 必须与网格形状 {u.shape} 相同")
    return b


def _pad_dirichlet(u, bnd):
    """给网格四周 pad 一层边界值，返回形状 +2 的数组（供差分取邻居）。

    仅内部区间对应的位置需要真实边界值；四角/棱的 pad 垫块不会被
    中心差分读到（每次只沿单一方向取一步邻居），填 0 即可。
    """
    k = u.ndim
    p = np.pad(u, 1, mode="constant", constant_values=0.0)
    for a in range(k):
        for face in (0, -1):
            src = [slice(None)] * k
            src[a] = face                      # bnd 的边界面，其余维全取
            dst = [slice(1, -1)] * k
            dst[a] = face                      # p 的对应面，其余维取内部区间
            p[tuple(dst)] = bnd[tuple(src)]
    return p


def _apply_boundary(u, bnd):
    """把 bnd 的值写到 u 的所有边界面（含棱与角，多次覆盖结果一致）。"""
    for a in range(u.ndim):
        for face in (0, -1):
            sl = [slice(None)] * u.ndim
            sl[a] = face
            u[tuple(sl)] = bnd[tuple(sl)]


def _laplacian(u, dxs, bnd):
    """中心差分 Laplacian ``Δu``（边界处无意义，调用方会覆盖边界面）。"""
    p = _pad_dirichlet(u, bnd)
    lap = np.zeros_like(u)
    for a in range(u.ndim):
        sl_lo = [slice(1, -1)] * u.ndim
        sl_lo[a] = slice(0, -2)
        sl_hi = [slice(1, -1)] * u.ndim
        sl_hi[a] = slice(2, None)
        lap += (p[tuple(sl_hi)] - 2.0 * u + p[tuple(sl_lo)]) / dxs[a] ** 2
    return lap


def _snapshot_indices(n_total, n_snapshots):
    """把 0..n_total 均匀抽成至多 n_snapshots 个（去重、保序、含首尾）。"""
    m = max(2, int(n_snapshots))
    return np.unique(np.linspace(0, n_total, min(m, n_total + 1)).astype(int))


def solve_heat_nd(u0, alpha, dx, dt, n_steps, boundary=0.0, n_snapshots=6):
    """n 维热传导方程 ``u_t = α·Δu`` 的显式 FTCS 格式（任意空间维数）。

    更新式 ``u^{n+1} = u^n + dt·α·Δ_h u^n``，其中 ``Δ_h`` 为 k 维中心差分
    Laplacian；整体精度 O(dt + dx²)。稳定性要求 ``Σ_i r_i ≤ 1/2``，
    其中 ``r_i = α·dt/dx_i²``。

    参数
    ----
    u0 : array_like, shape (n1, ..., nk)
        初值网格（每个维度都含边界节点）。
    alpha : float
        热扩散系数（> 0）。
    dx : float or sequence, length k
        各方向空间步长（标量表示所有方向相同）。
    dt : float
        时间步长（> 0）。
    n_steps : int
        演化步数（≥ 0）。
    boundary : float or ndarray, shape of u0
        Dirichlet 边界值（演化中保持不变）；数组时其边界面被采用，
        内部元素忽略。
    n_snapshots : int
        快照数量（含 t=0 与最终时刻，均匀抽样）。

    返回
    ----
    times : ndarray, shape (m,)
        各快照对应的时刻。
    snapshots : ndarray, shape (m, n1, ..., nk)
        快照序列。

    异常
    ----
    ValueError : 参数非法或违反稳定性条件时抛出。
    """
    u = _as_float_grid(u0)
    k = u.ndim
    dxs = _dx_sequence(dx, k)
    if alpha <= 0 or dt <= 0:
        raise ValueError("alpha、dt 必须为正数")
    if n_steps < 0:
        raise ValueError("n_steps 不能为负")

    r_sum = float(np.sum(alpha * dt / dxs**2))
    if r_sum > 0.5 + 1e-12:
        raise ValueError(
            f"FTCS 格式不稳定：Σ r_i = α·dt·Σ(1/dx_i²) = {r_sum:.4g} > 0.5，"
            "请减小 dt 或增大 dx（维数越高约束越紧）"
        )

    bnd = _resolve_boundary(boundary, u)
    idx = _snapshot_indices(n_steps, n_snapshots)
    snapshots = np.empty((idx.size, *u.shape))
    snapshots[0] = u

    cur = u.copy()
    si = 1
    for step in range(1, n_steps + 1):
        new = cur + alpha * dt * _laplacian(cur, dxs, bnd)
        _apply_boundary(new, bnd)
        if si < idx.size and step == idx[si]:
            snapshots[si] = new
            si += 1
        cur = new
    return idx.astype(float) * dt, snapshots


def solve_wave_nd(u0, v0, c, dx, dt, n_steps, boundary=0.0, n_snapshots=6):
    """n 维波动方程 ``u_tt = c²·Δu`` 的显式蛙跳格式（任意空间维数）。

    时间、空间均为二阶精度。第一层由泰勒展开构造；
    稳定性要求多维 CFL 条件 ``c·dt·sqrt(Σ_i 1/dx_i²) ≤ 1``。

    参数
    ----
    u0, v0 : array_like, shape (n1, ..., nk)
        初始位移与初始速度（同形状，含边界节点）。
    c : float
        波速（> 0）。
    dx : float or sequence, length k
        各方向空间步长。
    dt, n_steps, boundary, n_snapshots
        同 :func:`solve_heat_nd`。

    返回
    ----
    times : ndarray, shape (m,)
    snapshots : ndarray, shape (m, n1, ..., nk)

    异常
    ----
    ValueError : 参数非法或违反 CFL 条件时抛出。
    """
    u = _as_float_grid(u0)
    if np.isscalar(v0):
        v = np.full(u.shape, float(v0))
    else:
        v = _as_float_grid(v0, "v0")
    k = u.ndim
    if v.shape != u.shape:
        raise ValueError(f"v0 形状 {v.shape} 必须与 u0 形状 {u.shape} 相同")
    dxs = _dx_sequence(dx, k)
    if c <= 0 or dt <= 0:
        raise ValueError("c、dt 必须为正数")
    if n_steps < 0:
        raise ValueError("n_steps 不能为负")

    cfl = c * dt * float(np.sqrt(np.sum(1.0 / dxs**2)))
    if cfl > 1.0 + 1e-12:
        raise ValueError(
            f"违反 CFL 条件：c·dt·sqrt(Σ 1/dx_i²) = {cfl:.4g} > 1，"
            "请减小 dt 或增大 dx（维数越高约束越紧）"
        )

    bnd = _resolve_boundary(boundary, u)
    idx = _snapshot_indices(n_steps, n_snapshots)
    snapshots = np.empty((idx.size, *u.shape))
    snapshots[0] = u
    if n_steps == 0:
        return idx.astype(float) * dt, snapshots

    def _record(step, field):
        nonlocal si
        if si < idx.size and step == idx[si]:
            snapshots[si] = field
            si += 1

    si = 1
    # 第一层：u¹ = u⁰ + dt·v⁰ + ½(c·dt)²·Δu⁰
    prev = u.copy()
    cur = u + dt * v + 0.5 * (c * dt) ** 2 * _laplacian(u, dxs, bnd)
    _apply_boundary(cur, bnd)
    _record(1, cur)

    for step in range(2, n_steps + 1):
        new = 2.0 * cur - prev + (c * dt) ** 2 * _laplacian(cur, dxs, bnd)
        _apply_boundary(new, bnd)
        _record(step, new)
        prev, cur = cur, new
    return idx.astype(float) * dt, snapshots


def solve_laplace_nd(u_guess, dx=None, tol=1e-6, max_iter=20000, omega=1.0,
                     n_snapshots=6):
    """n 维（n ≥ 2）Laplace 方程 ``Δu = 0`` 的红黑 Gauss–Seidel/SOR 迭代。

    红黑（红黑棋盘着色）排序把内部节点按坐标和的奇偶分成两组交替更新，
    数学上与自然顺序 Gauss–Seidel 同阶收敛，但每组内互不依赖、可完全向量化，
    因此在高维网格上远快于逐点循环。``omega = 1`` 即红黑 Gauss–Seidel；
    ``1 < omega < 2`` 为 SOR 超松弛（可显著加速）。

    参数
    ----
    u_guess : ndarray, shape (n1, ..., nk), k ≥ 2
        边界元素给出 Dirichlet 边界条件（迭代中保持不变），
        内部元素作为迭代初值。
    dx : float or sequence, length k, optional
        各方向空间步长（仅影响不等距时的加权系数，默认等距）。
    tol : float
        相邻两次全迭代（红+黑两轮）的最大变化量小于 ``tol`` 时判定收敛。
    max_iter : int
        最大全迭代次数。
    omega : float
        松弛因子，须在 (0, 2) 内。
    n_snapshots : int
        快照数量（含初值与最终结果，按几何间隔抽样迭代次数）。

    返回
    ----
    snapshots : ndarray, shape (m, n1, ..., nk)
        快照序列（第 0 个为初值，最后一个为最终结果）。
    snap_iters : ndarray, shape (m,)
        各快照对应的迭代次数。

    异常
    ----
    ValueError : 网格维数/尺寸不足或 omega 非法时抛出。
    """
    u = _as_float_grid(u_guess, "u_guess")
    k = u.ndim
    if k < 2:
        raise ValueError("Laplace 方程至少需要 2 个空间维数（1 维退化为两点边值问题）")
    if not (0.0 < omega < 2.0):
        raise ValueError("omega 必须在 (0, 2) 内")
    if max_iter < 1 or tol <= 0:
        raise ValueError("max_iter 必须为正、tol 必须为正数")

    dxs = np.ones(k) if dx is None else _dx_sequence(dx, k)
    w_inv = 1.0 / dxs**2
    # nb 对每个方向求了左右两个邻居，故归一化分母是 2·Σ_a w_a（= 全部 2k 个
    # 邻居的权重和），漏乘 2 会把更新值放大约 2 倍并指数发散。
    w_sum = 2.0 * float(np.sum(w_inv))

    # 工作数组：u_work 含固定边界；未知数仅为去掉边界的内部区。
    # 每半轮（一个颜色）从当前 u_work 重建 pad 快照再更新，保证「另一颜色
    # 本轮已更新的新值立即参与本颜色计算」（红黑 Gauss–Seidel 语义）。
    u_work = u.copy()
    inner_u = tuple(slice(1, s - 1) for s in u.shape)          # u 的内部区
    color = np.indices(tuple(s - 2 for s in u.shape)).sum(axis=0) % 2

    cps = list(np.unique(np.rint(np.geomspace(1, max(2, max_iter),
                                              max(2, int(n_snapshots)))).astype(int)))
    snapshots = [u_work.copy()]
    snap_iters = [0]

    def _snap(marker):
        snapshots.append(u_work.copy())
        snap_iters.append(marker)

    it = 0
    while it < max_iter:
        diff = 0.0
        for parity in (0, 1):
            p = _pad_dirichlet(u_work, u_work)                 # 形状 +2
            mask = color == parity
            nb = np.zeros(color.shape, dtype=float)
            for a in range(k):
                sl_lo = [slice(2, -2)] * k
                sl_lo[a] = slice(1, -3)                        # 左邻居
                sl_hi = [slice(2, -2)] * k
                sl_hi[a] = slice(3, -1)                        # 右邻居
                nb += (p[tuple(sl_hi)] + p[tuple(sl_lo)]) * w_inv[a]
            inner = u_work[inner_u]                            # 视图，写入即写回
            upd = (1.0 - omega) * inner + omega * (nb / w_sum)
            diff = max(diff, float(np.max(np.abs(upd - inner))))
            inner[mask] = upd[mask]
        it += 1
        while cps and it >= cps[0]:
            cps.pop(0)
            _snap(it)
        if not np.isfinite(diff):
            # 注意 Python 的 max(0.0, nan) == 0.0，NaN 会被误判为已收敛，
            # 必须显式拦截
            raise ValueError(f"Gauss–Seidel 迭代出现非有限值（第 {it} 轮），请检查边界条件与网格")
        if diff < tol:
            break
    if not (snap_iters and snap_iters[-1] == it):
        _snap(it)

    snaps = np.stack(snapshots)
    return snaps, np.asarray(snap_iters, dtype=int)

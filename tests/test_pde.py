"""PDE 求解器单元测试：与精确解对比、稳定性守卫、线性代数正确性。"""

import numpy as np
import pytest

from diffeq.pde import (
    thomas_solve,
    solve_heat_ftcs,
    solve_heat_crank_nicolson,
    solve_wave,
    solve_laplace,
)


class TestThomas:
    def test_random_tridiagonal_system(self):
        rng = np.random.default_rng(42)
        n = 37
        sub = rng.normal(size=n - 1)
        diag = 4.0 + np.abs(rng.normal(size=n))  # 对角占优，保证稳定
        sup = rng.normal(size=n - 1)
        A = np.diag(diag) + np.diag(sub, -1) + np.diag(sup, 1)
        rhs = rng.normal(size=n)

        x = thomas_solve(sub, diag, sup, rhs)
        assert np.allclose(A @ x, rhs, atol=1e-10)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            thomas_solve([1.0], [2.0, 2.0], [1.0, 1.0], [1.0, 1.0])


def heat_exact(x, t):
    """零边界、初值 sin(pi*x) 的热传导方程精确解。"""
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)


class TestHeat:
    def test_ftcs_accuracy(self):
        nx = 200
        x = np.linspace(0.0, 1.0, nx + 1)
        dx = 1.0 / nx
        dt = 0.4 * dx**2          # r = 0.4 < 0.5
        n_steps = 1000            # T = 0.01
        u = solve_heat_ftcs(np.sin(np.pi * x), 1.0, dx, dt, n_steps)
        T = n_steps * dt
        assert u.shape == (n_steps + 1, nx + 1)
        assert np.max(np.abs(u[-1] - heat_exact(x, T))) < 5e-4

    def test_ftcs_unstable_raises(self):
        x = np.linspace(0.0, 1.0, 11)
        with pytest.raises(ValueError):
            solve_heat_ftcs(np.sin(np.pi * x), 1.0, 0.1, 0.1, 10)  # r = 10

    def test_crank_nicolson_accuracy_large_steps(self):
        nx = 100
        x = np.linspace(0.0, 1.0, nx + 1)
        dx = 1.0 / nx
        dt = 2.5e-3               # 时间步很大（r = 12.5），依赖无条件稳定性
        n_steps = 4               # T = 0.01
        u = solve_heat_crank_nicolson(np.sin(np.pi * x), 1.0, dx, dt, n_steps)
        assert np.max(np.abs(u[-1] - heat_exact(x, 0.01))) < 1e-3

    def test_crank_nicolson_conserves_boundary(self):
        x = np.linspace(0.0, 1.0, 21)
        u0 = np.sin(np.pi * x)
        u0[0], u0[-1] = 0.5, -0.3  # 非零 Dirichlet 边界
        u = solve_heat_crank_nicolson(u0, 1.0, 0.05, 1e-3, 50)
        assert u[-1, 0] == 0.5
        assert u[-1, -1] == -0.3


class TestWave:
    def test_standing_wave(self):
        nx, c, L = 200, 1.0, 1.0
        x = np.linspace(0.0, L, nx + 1)
        dx = L / nx
        dt = 0.004                 # CFL = 0.8
        n_steps = 500              # T = 2.0，精确解恰好回到初始位移
        u = solve_wave(np.sin(np.pi * x), np.zeros_like(x), c, dx, dt, n_steps)
        T = n_steps * dt
        exact = np.sin(np.pi * x) * np.cos(np.pi * c * T)
        assert u.shape == (n_steps + 1, nx + 1)
        assert np.max(np.abs(u[-1] - exact)) < 1e-3

    def test_cfl_violation_raises(self):
        x = np.zeros(11)
        with pytest.raises(ValueError):
            solve_wave(x, x, 2.0, 0.1, 0.1, 10)  # CFL = 2

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            solve_wave(np.zeros(11), np.zeros(10), 1.0, 0.1, 0.01, 10)


class TestLaplace:
    def test_linear_exact_solution(self):
        # 精确解 u = x*y 满足 Laplace 方程，离散解应收敛到节点精确值
        n = 20
        x = np.linspace(0.0, 1.0, n + 1)
        X, Y = np.meshgrid(x, x)
        guess = np.zeros((n + 1, n + 1))
        guess[0, :] = 0.0          # u(x, 0) = 0
        guess[-1, :] = x           # u(x, 1) = x
        guess[:, 0] = 0.0          # u(0, y) = 0
        guess[:, -1] = x           # u(1, y) = y

        u, n_iter = solve_laplace(guess, tol=1e-10, max_iter=50000)
        assert n_iter < 50000
        assert np.max(np.abs(u - X * Y)) < 1e-8

    def test_too_small_grid_raises(self):
        with pytest.raises(ValueError):
            solve_laplace(np.zeros((2, 2)))

"""n 维热传导 / 波动 / Laplace 求解器（``*_nd`` 系列）的单元测试。"""

import numpy as np
import pytest

from diffeq.pde import solve_heat_ftcs, solve_heat_nd, solve_laplace_nd, solve_wave_nd


def _mesh(n, k, L=1.0):
    """返回 [0, L]^k 上的均匀坐标网格（indexing='ij'，与网格轴顺序一致）。"""
    g = np.linspace(0.0, L, n)
    return np.meshgrid(*([g] * k), indexing="ij")


class TestHeatND:
    def test_2d_matches_exact(self):
        """初值 sin(πx)sin(πy) 的精确解为 e^{-2π²t}·sin·sin。"""
        n = 41
        dx = 1.0 / (n - 1)
        X, Y = _mesh(n, 2)
        u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
        dt = 0.4 * dx**2 / 2.0          # Σ r_i = 0.4 ≤ 1/2
        times, snaps = solve_heat_nd(u0, 1.0, dx, dt, 40, boundary=0.0,
                                     n_snapshots=5)
        assert snaps.shape == (5, n, n)
        assert np.all(np.diff(times) > 0)
        exact = np.exp(-2.0 * np.pi**2 * times[-1]) * u0
        assert np.max(np.abs(snaps[-1] - exact)) < 5e-4

    def test_3d_smoke(self):
        n = 17
        dx = 1.0 / (n - 1)
        X, Y, Z = _mesh(n, 3)
        u0 = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
        dt = 0.3 / (3.0 * (n - 1) ** 2)
        times, snaps = solve_heat_nd(u0, 1.0, dx, dt, 30, n_snapshots=4)
        assert snaps.shape == (4, n, n, n)
        exact = np.exp(-3.0 * np.pi**2 * times[-1]) * u0
        assert np.max(np.abs(snaps[-1] - exact)) < 5e-3

    def test_1d_agrees_with_legacy_ftcs(self):
        u1 = np.sin(np.pi * np.linspace(0.0, 1.0, 51))
        _, s_nd = solve_heat_nd(u1, 1.0, 0.02, 1e-4, 200, n_snapshots=3)
        s_1d = solve_heat_ftcs(u1, 1.0, 0.02, 1e-4, 200)
        assert np.max(np.abs(s_nd[-1] - s_1d[-1])) < 1e-12

    def test_boundary_held_fixed(self):
        n = 11
        X, _ = _mesh(n, 2)
        # 初值内部为 1，边界与零 Dirichlet 条件保持一致
        u0 = np.ones((n, n))
        u0[0, :], u0[-1, :], u0[:, 0], u0[:, -1] = 0.0, 0.0, 0.0, 0.0
        # dx=0.1 时稳定性要求 dt ≤ 0.25·dx² = 0.0025
        _, snaps = solve_heat_nd(u0, 1.0, 0.1, 0.002, 25, boundary=0.0)
        # 零 Dirichlet 边界在整个演化中保持为 0
        assert np.all(snaps[:, 0, :] == 0.0)
        assert np.all(snaps[:, -1, :] == 0.0)
        assert np.all(snaps[:, :, 0] == 0.0)
        assert np.all(snaps[:, :, -1] == 0.0)
        # 内部热量向外扩散，最大值单调不增
        peaks = np.max(np.max(snaps, axis=2), axis=1)
        assert np.all(np.diff(peaks) <= 1e-12)

    def test_unstable_raises(self):
        n = 21
        X, Y = _mesh(n, 2)
        u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
        with pytest.raises(ValueError, match="不稳定"):
            solve_heat_nd(u0, 1.0, 0.05, 0.05**2, 5)   # Σ r = 2 > 1/2

    def test_bad_inputs_raise(self):
        X, Y = _mesh(11, 2)
        u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
        with pytest.raises(ValueError):
            solve_heat_nd(u0, -1.0, 0.1, 0.001, 5)     # alpha < 0
        with pytest.raises(ValueError):
            solve_heat_nd(u0, 1.0, [0.1, 0.1, 0.1], 0.001, 5)  # dx 长度不符


class TestWaveND:
    def test_2d_standing_wave(self):
        """初速为零、初值 sin(πx)sin(πy) 的精确解为 cos(√2·π·t)·初值。"""
        n = 61
        dx = 1.0 / (n - 1)
        X, Y = _mesh(n, 2)
        u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
        dt = 0.9 / (np.sqrt(2.0) * (n - 1))             # CFL ≈ 0.9
        times, snaps = solve_wave_nd(u0, 0.0, 1.0, dx, dt, 80, n_snapshots=3)
        assert snaps.shape == (3, n, n)
        exact = np.cos(np.sqrt(2.0) * np.pi * times[-1]) * u0
        assert np.max(np.abs(snaps[-1] - exact)) < 5e-4

    def test_cfl_raises(self):
        n = 21
        X, Y = _mesh(n, 2)
        u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
        with pytest.raises(ValueError, match="CFL"):
            solve_wave_nd(u0, 0.0, 1.0, 0.05, 0.1, 5)   # CFL ≈ 2 > 1

    def test_scalar_v0_broadcast(self):
        n = 21
        X, Y = _mesh(n, 2)
        u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
        _, s1 = solve_wave_nd(u0, 0.0, 1.0, 0.05, 0.03, 10, n_snapshots=2)
        v0 = np.zeros_like(u0)
        _, s2 = solve_wave_nd(u0, v0, 1.0, 0.05, 0.03, 10, n_snapshots=2)
        assert np.array_equal(s1[-1], s2[-1])

    def test_zero_steps_returns_initial(self):
        n = 11
        X, Y = _mesh(n, 2)
        u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
        times, snaps = solve_wave_nd(u0, 0.0, 1.0, 0.1, 0.05, 0, n_snapshots=3)
        assert snaps.shape == (1, n, n)
        assert np.array_equal(snaps[0], u0)


class TestLaplaceND:
    @staticmethod
    def _harmonic_grid(n, k):
        """x₀² − x₁² 是调和函数（与其余坐标无关），用作解析解。"""
        grids = _mesh(n, k)
        exact = grids[0] ** 2 - grids[1] ** 2
        bnd_mask = np.zeros((n,) * k, dtype=bool)
        for a in range(k):
            for face in (0, -1):
                sl = [slice(None)] * k
                sl[a] = face
                bnd_mask[tuple(sl)] = True
        u_guess = np.where(bnd_mask, exact, 0.0)
        return u_guess, exact

    def test_2d_harmonic(self):
        u_guess, exact = self._harmonic_grid(21, 2)
        snaps, iters = solve_laplace_nd(u_guess, tol=1e-8, max_iter=30000,
                                        n_snapshots=4)
        assert snaps.shape == (4, 21, 21)
        assert 0 < iters[-1] <= 30000
        assert np.max(np.abs(snaps[-1] - exact)) < 1e-6

    def test_3d_harmonic(self):
        u_guess, exact = self._harmonic_grid(13, 3)
        snaps, iters = solve_laplace_nd(u_guess, tol=1e-8, max_iter=50000)
        assert snaps.shape[1:] == (13, 13, 13)
        assert np.max(np.abs(snaps[-1] - exact)) < 1e-6

    def test_boundary_held_fixed(self):
        u_guess, exact = self._harmonic_grid(11, 2)
        snaps, _ = solve_laplace_nd(u_guess, tol=1e-6, max_iter=5000)
        assert np.allclose(snaps[-1][0, :], exact[0, :])
        assert np.allclose(snaps[-1][:, -1], exact[:, -1])

    def test_sor_converges_no_slower(self):
        u_guess, exact = self._harmonic_grid(21, 2)
        _, it_gs = solve_laplace_nd(u_guess, tol=1e-8, max_iter=30000, omega=1.0)
        _, it_sor = solve_laplace_nd(u_guess, tol=1e-8, max_iter=30000, omega=1.8)
        assert it_sor[-1] <= it_gs[-1]

    def test_bad_inputs_raise(self):
        u_guess, _ = self._harmonic_grid(11, 2)
        with pytest.raises(ValueError, match="2 个空间维数"):
            solve_laplace_nd(np.zeros(9))
        with pytest.raises(ValueError, match="omega"):
            solve_laplace_nd(u_guess, omega=2.5)
        with pytest.raises(ValueError, match="omega"):
            solve_laplace_nd(u_guess, omega=0.0)

"""ODE 求解器单元测试：精度、输出形状、收敛阶与自适应步长。"""

import numpy as np
import pytest

from diffeq.ode import solve_ode, solve_ivp
from diffeq.diagnostics import convergence_orders, endpoint_error

K = 0.5


def f_decay(t, y):
    return -K * y


def exact_decay(t):
    return 2.0 * np.exp(-K * t)


def f_harmonic(t, s):
    """谐振子方程组 x'' + x = 0 化为一阶方程组 s = [x, v]。"""
    x, v = s
    return np.array([v, -x])


class TestFixedStep:
    @pytest.mark.parametrize(
        "method,tol", [("euler", 5e-2), ("heun", 1e-3), ("rk4", 1e-9)]
    )
    def test_decay_endpoint_accuracy(self, method, tol):
        t, y = solve_ode(f_decay, 2.0, (0.0, 4.0), n_steps=400, method=method)
        assert len(t) == 401
        assert abs(t[-1] - 4.0) < 1e-12
        assert abs(y[-1] - exact_decay(4.0)) < tol

    def test_rk4_system_harmonic(self):
        t, y = solve_ode(f_harmonic, [1.0, 0.0], (0.0, 2 * np.pi), n_steps=400)
        assert y.shape == (401, 2)
        assert abs(y[-1, 0] - 1.0) < 1e-7
        assert abs(y[-1, 1]) < 1e-7
        # 能量 x^2 + v^2 应在数值精度内守恒
        energy = y[:, 0] ** 2 + y[:, 1] ** 2
        assert np.max(np.abs(energy - 1.0)) < 1e-6

    def test_scalar_output_shape(self):
        t, y = solve_ode(f_decay, 2.0, (0.0, 1.0), n_steps=10)
        assert y.shape == (11,)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            solve_ode(f_decay, 2.0, (0.0, 1.0), method="magic")

    def test_bad_n_steps_raises(self):
        with pytest.raises(ValueError):
            solve_ode(f_decay, 2.0, (0.0, 1.0), n_steps=0)


class TestConvergenceOrder:
    """用收敛阶估计验证方法的真实精度阶。"""

    def test_rk4_is_fourth_order(self):
        orders = convergence_orders(
            f_decay, 2.0, (0.0, 1.0), "rk4", exact_decay, [10, 20, 40, 80]
        )
        assert 3.7 < np.mean(orders) < 4.3

    def test_heun_is_second_order(self):
        orders = convergence_orders(
            f_decay, 2.0, (0.0, 1.0), "heun", exact_decay, [20, 40, 80, 160]
        )
        assert 1.8 < np.mean(orders) < 2.2

    def test_euler_is_first_order(self):
        orders = convergence_orders(
            f_decay, 2.0, (0.0, 1.0), "euler", exact_decay, [50, 100, 200, 400]
        )
        assert 0.9 < np.mean(orders) < 1.1

    def test_endpoint_error_decreases(self):
        e1 = endpoint_error(f_decay, 2.0, (0.0, 1.0), "rk4", 10, exact_decay)
        e2 = endpoint_error(f_decay, 2.0, (0.0, 1.0), "rk4", 20, exact_decay)
        assert e2 < e1


class TestAdaptiveRK45:
    def test_decay(self):
        t, y = solve_ivp(f_decay, 2.0, (0.0, 4.0), rtol=1e-8, atol=1e-10)
        assert np.all(np.diff(t) > 0)
        assert abs(t[-1] - 4.0) < 1e-12
        assert abs(y[-1] - exact_decay(4.0)) < 1e-6

    def test_system(self):
        t, y = solve_ivp(f_harmonic, [1.0, 0.0], (0.0, 2 * np.pi),
                         rtol=1e-9, atol=1e-12)
        assert y.shape[1] == 2
        assert abs(y[-1, 0] - 1.0) < 1e-7

    def test_scalar_output_shape(self):
        t, y = solve_ivp(f_decay, 2.0, (0.0, 1.0))
        assert y.shape == (len(t),)

    def test_first_step_respected(self):
        t, _ = solve_ivp(f_decay, 2.0, (0.0, 1.0), first_step=0.1)
        assert abs(t[1] - t[0] - 0.1) < 1e-12

"""diffeq-toolkit —— 常微分方程（ODE）与偏微分方程（PDE）数值求解工具库。

模块结构
--------
- ``diffeq.ode``         : 常微分方程初值问题求解（Euler / Heun / RK4 / 自适应 RK45）
- ``diffeq.pde``         : 偏微分方程数值解（热传导、波动、Laplace 方程）
- ``diffeq.diagnostics`` : 数值诊断（端点误差、观测收敛阶估计）
- ``diffeq.plotting``    : 基于 matplotlib 的可视化辅助函数
"""

from .ode import euler_step, heun_step, rk4_step, solve_ode, solve_ivp
from .pde import (
    thomas_solve,
    solve_heat_ftcs,
    solve_heat_crank_nicolson,
    solve_wave,
    solve_laplace,
    solve_heat_nd,
    solve_wave_nd,
    solve_laplace_nd,
)
from .diagnostics import endpoint_error, convergence_orders
from . import plotting

__version__ = "0.1.0"

__all__ = [
    "euler_step",
    "heun_step",
    "rk4_step",
    "solve_ode",
    "solve_ivp",
    "thomas_solve",
    "solve_heat_ftcs",
    "solve_heat_crank_nicolson",
    "solve_wave",
    "solve_laplace",
    "solve_heat_nd",
    "solve_wave_nd",
    "solve_laplace_nd",
    "endpoint_error",
    "convergence_orders",
    "plotting",
    "__version__",
]

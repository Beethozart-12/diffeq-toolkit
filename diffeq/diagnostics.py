"""数值诊断工具：端点误差计算与观测收敛阶估计。"""

from __future__ import annotations

import numpy as np

from .ode import solve_ode

__all__ = ["endpoint_error", "convergence_orders"]


def endpoint_error(f, y0, t_span, method, n_steps, y_exact):
    """计算数值解在 ``t_span`` 终点处相对精确解的最大绝对误差。

    参数
    ----
    f, y0, t_span, method, n_steps :
        含义同 :func:`diffeq.ode.solve_ode`。
    y_exact : callable
        精确解 ``y_exact(t)``（标量或与状态同形状的数组）。

    返回
    ----
    float
        终点误差 ``max_i |y_num_i - y_exact_i|``。
    """
    t, y = solve_ode(f, y0, t_span, n_steps=n_steps, method=method)
    exact = np.asarray(y_exact(t[-1]), dtype=float).reshape(-1)
    numeric = np.asarray(y[-1], dtype=float).reshape(-1)
    return float(np.max(np.abs(numeric - exact)))


def convergence_orders(f, y0, t_span, method, y_exact, n_steps_list):
    """用步数序列估计数值方法的观测收敛阶。

    对相邻两档步数按 ``p = log(E1/E2) / log(n2/n1)`` 计算观测阶
    （``E`` 为终点误差，``n`` 为步数，``h ∝ 1/n``）。

    参数
    ----
    f, y0, t_span, method :
        含义同 :func:`diffeq.ode.solve_ode`。
    y_exact : callable
        精确解 ``y_exact(t)``。
    n_steps_list : sequence of int
        步数序列（至少 2 个，建议等比递增）。

    返回
    ----
    list[float]
        每相邻两档步数对应的观测阶，长度为 ``len(n_steps_list) - 1``。
    """
    n_steps_list = [int(n) for n in n_steps_list]
    if len(n_steps_list) < 2:
        raise ValueError("n_steps_list 至少需要两个元素")

    errors = [endpoint_error(f, y0, t_span, method, n, y_exact) for n in n_steps_list]
    orders = []
    for e1, e2, n1, n2 in zip(
        errors[:-1], errors[1:], n_steps_list[:-1], n_steps_list[1:]
    ):
        if e1 <= 0 or e2 <= 0:
            raise ValueError(
                f"终点误差为 0（e1={e1}, e2={e2}），无法估计收敛阶；"
                "请选用更精细的参考解或更小的步长范围"
            )
        orders.append(float(np.log(e1 / e2) / np.log(n2 / n1)))
    return orders

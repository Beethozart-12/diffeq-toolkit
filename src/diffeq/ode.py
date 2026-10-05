"""常微分方程（ODE）初值问题数值求解模块。

支持标量方程与一阶方程组 ``dy/dt = f(t, y)``：

- 定步长方法：显式 Euler（一阶）、Heun / 改进 Euler（二阶）、经典 RK4（四阶）
- 自适应步长方法：Dormand–Prince RK45（5(4) 嵌入对，误差自动控制）
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "euler_step",
    "heun_step",
    "rk4_step",
    "solve_ode",
    "solve_ivp",
]


# ----------------------------------------------------------------------
# 定步长单步格式
# ----------------------------------------------------------------------

def _as_vector(y0):
    """把标量/任意形状初值统一为一维 float 数组。"""
    return np.asarray(y0, dtype=float).reshape(-1)


def euler_step(f, t, y, h):
    """显式 Euler 单步推进 ``y_{n+1} = y_n + h·f(t_n, y_n)``。整体一阶精度。"""
    return y + h * np.asarray(f(t, y), dtype=float)


def heun_step(f, t, y, h):
    """Heun（改进 Euler，二阶 Runge–Kutta）单步推进。整体二阶精度。"""
    k1 = np.asarray(f(t, y), dtype=float)
    k2 = np.asarray(f(t + h, y + h * k1), dtype=float)
    return y + 0.5 * h * (k1 + k2)


def rk4_step(f, t, y, h):
    """经典四阶 Runge–Kutta 单步推进。整体四阶精度。"""
    k1 = np.asarray(f(t, y), dtype=float)
    k2 = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k1), dtype=float)
    k3 = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k2), dtype=float)
    k4 = np.asarray(f(t + h, y + h * k3), dtype=float)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


_STEP_METHODS = {"euler": euler_step, "heun": heun_step, "rk4": rk4_step}


# ----------------------------------------------------------------------
# 定步长驱动器
# ----------------------------------------------------------------------

def solve_ode(f, y0, t_span, n_steps=200, method="rk4"):
    """用定步长方法求解初值问题 ``dy/dt = f(t, y),  y(t0) = y0``。

    参数
    ----
    f : callable
        右端函数 ``f(t, y)``，返回与 y 同形状的导数（标量或 ndarray）。
    y0 : float 或 array_like
        初始状态。
    t_span : (t0, tf)
        求解区间。
    n_steps : int
        演化步数，步长 ``h = (tf - t0) / n_steps``。
    method : {"euler", "heun", "rk4"}
        求解方法，默认 "rk4"。

    返回
    ----
    t : ndarray, shape (n_steps+1,)
        时间网格。
    y : ndarray
        标量问题返回 shape ``(n_steps+1,)``；方程组返回 shape ``(n_steps+1, n)``。

    异常
    ----
    ValueError : 方法未知或 ``n_steps`` 非法时抛出。
    """
    if method not in _STEP_METHODS:
        raise ValueError(f"未知方法 {method!r}，可选：{sorted(_STEP_METHODS)}")
    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError("n_steps 必须为正整数")

    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n_steps
    step = _STEP_METHODS[method]

    y = _as_vector(y0)
    was_scalar = np.asarray(y0).ndim == 0
    ts = np.linspace(t0, tf, n_steps + 1)
    ys = np.empty((n_steps + 1, y.size))
    ys[0] = y
    for i in range(n_steps):
        y = step(f, ts[i], y, h)
        ys[i + 1] = y
    return ts, (ys[:, 0] if was_scalar else ys)


# ----------------------------------------------------------------------
# 自适应 Dormand–Prince RK45（5(4) 嵌入对）
# ----------------------------------------------------------------------

_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_B5 = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0])
_B4 = np.array([5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])


def _dopri5_step(f, t, y, h):
    """Dormand–Prince 单步：返回五阶解与嵌入四阶解。"""
    k = np.empty((7, y.size))
    for i in range(7):
        acc = np.zeros_like(y)
        for j in range(i):
            acc += _A[i][j] * k[j]
        k[i] = np.asarray(f(t + _C[i] * h, y + h * acc), dtype=float)
    y5 = y + h * (_B5 @ k)
    y4 = y + h * (_B4 @ k)
    return y5, y4


def _initial_step(f, t0, y, span, rtol, atol):
    """Hairer 启发式：估计合适的初始步长。"""
    scale = atol + rtol * np.abs(y)
    d0 = np.sqrt(np.mean((y / scale) ** 2))
    f0 = np.asarray(f(t0, y), dtype=float)
    d1 = np.sqrt(np.mean((f0 / scale) ** 2))
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    h0 = min(h0, abs(span))
    h0 = h0 if h0 > 0 else 1e-6

    y1 = y + h0 * f0
    f1 = np.asarray(f(t0 + h0, y1), dtype=float)
    scale1 = atol + rtol * np.abs(y1)
    d2 = np.sqrt(np.mean(((f1 - f0) / scale1) ** 2)) / h0
    if max(d1, d2) <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** 0.2
    return min(100.0 * h0, h1, abs(span))


def solve_ivp(f, y0, t_span, rtol=1e-6, atol=1e-9, first_step=None, max_steps=100000):
    """自适应 Dormand–Prince 5(4)（RK45）求解初值问题（类似 MATLAB ode45）。

    每步计算五阶解与嵌入四阶解之差作为局部误差估计，
    按误差自动放大/缩小步长，使加权 RMS 误差不超过 1：

        scale_i = atol + rtol · max(|y_i|, |y_new_i|)
        err = rms((y5 - y4) / scale)  ≤ 1

    参数
    ----
    f : callable
        右端函数 ``f(t, y)``。
    y0 : float 或 array_like
        初始状态。
    t_span : (t0, tf)
        求解区间（``tf < t0`` 时向反向积分）。
    rtol, atol : float
        相对与绝对容差。
    first_step : float, 可选
        初始步长；缺省时自动估计。
    max_steps : int
        最大步数，防止死循环。

    返回
    ----
    t : ndarray, shape (n,)
    y : ndarray
        标量问题 shape ``(n,)``；方程组 shape ``(n, m)``。

    异常
    ----
    RuntimeError : 步长塌缩或超过最大步数（常见于刚性方程）时抛出。
    """
    t0, tf = float(t_span[0]), float(t_span[1])
    y = _as_vector(y0)
    was_scalar = np.asarray(y0).ndim == 0

    span = abs(tf - t0)
    h = min(abs(float(first_step)), span) if first_step is not None \
        else _initial_step(f, t0, y, span, rtol, atol)
    direction = 1.0 if tf >= t0 else -1.0

    ts = [t0]
    ys = [y.copy()]
    t = t0

    for _ in range(int(max_steps)):
        if abs(tf - t) <= 1e-14 * max(1.0, abs(t)):
            break
        h = min(h, abs(tf - t))
        h_signed = direction * h

        y_new, y_err = _dopri5_step(f, t, y, h_signed)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err = np.sqrt(np.mean(((y_new - y_err) / scale) ** 2))

        if not np.isfinite(err):
            err = np.inf

        if err <= 1.0:  # 接受该步
            t = t + h_signed
            y = y_new
            ts.append(t)
            ys.append(y.copy())
            factor = 5.0 if err == 0.0 else 0.9 * err ** (-0.2)
            h *= min(5.0, max(0.2, factor))
        else:           # 拒绝并缩步
            h *= max(0.1, 0.9 * min(err, 1e10) ** (-0.2))

        if not np.isfinite(h) or h <= 1e-15:
            raise RuntimeError(
                "步长塌缩，求解失败：方程可能呈刚性行为，请放大容差或改用隐式方法"
            )
    else:
        raise RuntimeError(f"达到最大步数 {max_steps} 仍未到达终点，请放宽容差")

    t_arr = np.asarray(ts)
    y_arr = np.asarray(ys)
    return t_arr, (y_arr[:, 0] if was_scalar else y_arr)

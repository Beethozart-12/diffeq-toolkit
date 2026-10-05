"""基于 matplotlib 的可视化辅助函数。

所有函数均返回 ``(fig, ax)``，便于进一步定制；
``save`` 参数可直接把图片写入文件，``show`` 参数控制是否弹出窗口。
"""

from __future__ import annotations

import numpy as np
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt

__all__ = ["setup_cjk_font", "plot_ode", "plot_heat", "plot_snapshots", "plot_laplace"]

# 常见中文字体（按优先级），找到任一即启用，避免图中文本出现方框
_CJK_CANDIDATES = (
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
)


def setup_cjk_font() -> None:
    """若系统存在中文字体则设为 matplotlib 首选无衬线字体。

    在模块导入时自动调用；使用原生 ``matplotlib.pyplot`` 绘图的
    用户脚本也可手动调用以获得相同的中文显示效果。
    """
    try:
        available = {f.name for f in _fm.fontManager.ttflist}
    except Exception:  # 字体缓存异常时跳过，不影响绘图主功能
        return
    for name in _CJK_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


setup_cjk_font()


def _finalize(fig, save=None, show=False):
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


def plot_ode(t, y, labels=None, title="ODE 数值解", xlabel="$t$", ylabel="$y$",
             save=None, show=False, ax=None):
    """绘制 ODE 数值解曲线。

    参数
    ----
    t : array_like, shape (n,)
    y : array_like
        shape ``(n,)`` 或 ``(n, m)``（多状态分量按列绘制多条曲线）。
    labels : list[str], 可选
        每条曲线的图例文字。

    返回
    ----
    (fig, ax)
    """
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    else:
        fig = ax.figure

    y_arr = np.asarray(y)
    if y_arr.ndim == 1:
        y_arr = y_arr[:, None]
    for j in range(y_arr.shape[1]):
        if labels and j < len(labels):
            label = labels[j]
        else:
            label = "y" if y_arr.shape[1] == 1 else f"$y_{j}$"
        ax.plot(t, y_arr[:, j], label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _finalize(fig, save, show)
    return fig, ax


def plot_heat(u, x, t, title="热传导方程数值解", save=None, show=False,
              cmap="inferno", ax=None):
    """绘制热传导方程解的时空热图。

    参数
    ----
    u : array_like, shape (n_time, n_x)
        解历史（第 k 行为第 k 个时刻）。
    x : array_like, shape (n_x,)
    t : array_like, shape (n_time,)

    返回
    ----
    (fig, ax)
    """
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    else:
        fig = ax.figure

    mesh = ax.pcolormesh(x, t, np.asarray(u), shading="auto", cmap=cmap)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="$u(x, t)$")
    _finalize(fig, save, show)
    return fig, ax


def plot_snapshots(u, x, t, n=5, title="解的空间分布快照", save=None, show=False,
                   ax=None):
    """在等间隔选取的 ``n`` 个时刻绘制空间分布快照曲线。

    参数
    ----
    u : array_like, shape (n_time, n_x)
    x : array_like, shape (n_x,)
    t : array_like, shape (n_time,)

    返回
    ----
    (fig, ax)
    """
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    else:
        fig = ax.figure

    u = np.asarray(u)
    idx = np.linspace(0, u.shape[0] - 1, min(n, u.shape[0])).astype(int)
    for i in idx:
        ax.plot(x, u[i], label=f"$t = {t[i]:.3g}$")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$u$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _finalize(fig, save, show)
    return fig, ax


def plot_laplace(u, x=None, y=None, title="Laplace 方程数值解", save=None,
                 show=False, cmap="viridis", ax=None):
    """绘制二维 Laplace 方程的解（热图 + 等值线）。

    参数
    ----
    u : array_like, shape (ny+1, nx+1)
    x, y : array_like, 可选
        两个方向的坐标网格（缺省用节点序号）。

    返回
    ----
    (fig, ax)
    """
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
    else:
        fig = ax.figure

    u = np.asarray(u)
    if x is None:
        x = np.arange(u.shape[1])
    if y is None:
        y = np.arange(u.shape[0])

    im = ax.imshow(u, origin="lower", aspect="auto", cmap=cmap,
                   extent=[float(np.min(x)), float(np.max(x)),
                           float(np.min(y)), float(np.max(y))])
    cs = ax.contour(x, y, u, colors="white", linewidths=0.6, alpha=0.7)
    ax.clabel(cs, fmt="%.2f", fontsize=8)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="$u$")
    _finalize(fig, save, show)
    return fig, ax

"""绘图模块冒烟测试（无头环境，Agg 后端）。"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from diffeq.plotting import plot_ode, plot_heat, plot_snapshots, plot_laplace


def test_plot_ode_single_and_multi():
    t = np.linspace(0.0, 1.0, 11)
    fig, ax = plot_ode(t, np.sin(t), title="single")
    assert fig is not None and ax is not None
    plt.close(fig)

    fig, ax = plot_ode(t, np.column_stack([np.sin(t), np.cos(t)]),
                       labels=["sin", "cos"], save=None)
    assert len(ax.lines) == 2
    plt.close(fig)


def test_plot_heat():
    x = np.linspace(0, 1, 21)
    t = np.linspace(0, 1, 11)
    u = np.sin(np.pi * x)[None, :] * np.exp(-np.pi**2 * t)[:, None]
    fig, ax = plot_heat(u, x, t)
    assert ax.collections, "热图应包含 pcolormesh 集合"
    plt.close(fig)


def test_plot_snapshots():
    x = np.linspace(0, 1, 21)
    t = np.linspace(0, 1, 11)
    u = np.tile(np.sin(np.pi * x), (11, 1))
    fig, ax = plot_snapshots(u, x, t, n=4)
    assert len(ax.lines) == 4
    plt.close(fig)


def test_plot_laplace():
    n = 10
    x = np.linspace(0, 1, n + 1)
    X, Y = np.meshgrid(x, x)
    fig, ax = plot_laplace(X * Y, x, x)
    assert ax.collections, "等值线应存在"
    plt.close(fig)

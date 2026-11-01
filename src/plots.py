import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.ticker import FuncFormatter, LogLocator, PercentFormatter

from .backtest import LOOKBACKS
from .metrics import wealth
from .optimization import efficient_frontier, min_variance, optimal_portfolio

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LOOKBACK_COLORS = dict(zip(LOOKBACKS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
HALO = [patheffects.withStroke(linewidth=3, foreground=SURFACE)]
OUTLINE = [patheffects.withStroke(linewidth=1.5, foreground=INK_SECONDARY)]

FAMILIES = {"MinVar": "Minimum variance", "Optimal": "Optimal portfolio"}
BENCHMARKS = {"1/N": "Equal weight (1/N)", "Market": "CRSP market"}
TABLE_COLUMNS = {  # column: (header, format)
    "CAGR": ("CAGR", "{:.1%}"),
    "Volatility": ("Volatility", "{:.1%}"),
    "Sharpe": ("Sharpe\nratio", "{:.2f}"),
    "Max drawdown": ("Max\ndrawdown", "{:.1%}"),
    "Alpha": ("CAPM\nalpha", "{:.1%}"),
    "Alpha p-value": ("Alpha\np-value", "{:.3f}"),
    "Beta": ("Beta", "{:.2f}"),
}


def _display_name(name):
    if name in BENCHMARKS:
        return BENCHMARKS[name]
    family, lookback = name.split()
    return f"{FAMILIES[family]} {lookback}"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.title.set_color(INK)


def plot_wealth(returns, family, path=None):
    """Growth of $1 (log scale) for one strategy family against 1/N and the market.
    The y-axis spans all strategies so that both families share the same scale."""
    w = wealth(returns)
    start = w.index[0] - 1
    w = pd.concat([pd.DataFrame(1.0, index=[start], columns=w.columns), w])
    w.index = w.index.to_timestamp(how="end")

    fig, ax = plt.subplots(figsize=(10, 6))
    _style(ax)
    for L in LOOKBACKS:
        ax.plot(w[f"{family} {L}y"], color=LOOKBACK_COLORS[L], lw=1.5, label=f"{FAMILIES[family]} {L}y")
    ax.plot(w["1/N"], color=MUTED, lw=1.5, ls="--", label="Equal weight (1/N)")
    ax.plot(w["Market"], color=INK, lw=1.5, label="CRSP market")

    ax.set_yscale("log")
    ax.set_ylim(w.min().min() * 0.9, w.max().max() * 1.1)
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y:,.0f}"))
    ax.yaxis.set_minor_locator(LogLocator(subs=()))
    ax.margins(x=0.01)
    ax.set_title(f"{FAMILIES[family]} vs Benchmarks", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK_SECONDARY)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


def _contrast(color, background=SURFACE):
    """WCAG contrast ratio between two colours."""
    def luminance(c):
        c = np.array(plt.matplotlib.colors.to_rgb(c))
        c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return c @ [0.2126, 0.7152, 0.0722]
    lighter, darker = sorted([luminance(color), luminance(background)], reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _place_labels(ax, points, labels, colors, avoid, taken):
    """Greedy placement: each label goes to the first offset around its point whose box overlaps
    neither previous labels (`taken`, display boxes) nor other points (`avoid`, display coords).
    Labels too light to read on the background get a dark outline."""
    renderer = ax.figure.canvas.get_renderer()
    offsets = [(6, 5), (6, -5), (-6, 5), (-6, -5), (0, 11), (0, -11), (10, 0), (-10, 0)]
    for (x, y), label, color in zip(points, labels, colors):
        effects = OUTLINE if _contrast(color) < 1.5 else HALO
        best = None
        for dx, dy in offsets:
            ha = "left" if dx > 0 else "right" if dx < 0 else "center"
            text = ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                               va="center", fontsize=8, color=color, path_effects=effects, zorder=6)
            box = text.get_window_extent(renderer).padded(2)
            clashes = sum(box.overlaps(b) for b in taken) + sum(box.contains(px, py) for px, py in avoid)
            if best is None or clashes < best[0]:
                if best:
                    best[1].remove()
                best = (clashes, text, box)
            else:
                text.remove()
            if clashes == 0:
                break
        taken.append(best[2])


def plot_frontier(daily, monthly, rf, rebalance_dates, lookback=10, n_labels=20, path=None):
    """Long-only efficient frontier at the last rebalancing date, with the 49 industries coloured
    by Sharpe ratio, the minimum variance, optimal and 1/N portfolios and the Capital Market Line.
    Labels the n_labels/2 industries with the highest and the lowest Sharpe ratio."""
    t, m = rebalance_dates.iloc[-1], rebalance_dates.index[-1]
    cov = daily.loc[t - pd.DateOffset(years=lookback) + pd.Timedelta(days=1): t].cov().values * 252
    mu = monthly.loc[m - 12 * lookback + 1: m].mean().values * 12
    r_f = rf.loc[m - 12 * lookback + 1: m].mean() * 12
    n = len(mu)

    def stats(W):
        W = np.atleast_2d(W)
        return np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W)), W @ mu

    _, W_frontier = efficient_frontier(cov, mu)
    vol_f, ret_f = stats(W_frontier)
    portfolios = {  # weights, marker, label offset (points)
        "Minimum variance": (min_variance(cov), "s", (-10, 0)),
        "Optimal portfolio": (optimal_portfolio(cov, mu - r_f), "*", (-10, 6)),
        "Equal weight (1/N)": (np.full(n, 1 / n), "D", (-10, 0)),
    }
    vol_o, ret_o = stats(portfolios["Optimal portfolio"][0])
    vol_mv, ret_mv = stats(portfolios["Minimum variance"][0])
    vol_i, ret_i = np.sqrt(np.diag(cov)), mu
    sharpe_i = (ret_i - r_f) / vol_i

    fig, ax = plt.subplots(figsize=(10, 6))
    _style(ax)

    sc = ax.scatter(vol_i, ret_i, c=sharpe_i, cmap="viridis", s=45, edgecolor=SURFACE, lw=0.8, zorder=2)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Sharpe ratio", color=INK_SECONDARY)
    cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    cbar.outline.set_visible(False)

    efficient = ret_f >= ret_mv[0]
    ax.plot(vol_f[efficient], ret_f[efficient], color=INK, lw=2, zorder=3, label="Efficient frontier")
    ax.plot(vol_f[~efficient], ret_f[~efficient], color=INK, lw=0.8, ls="--", zorder=3)

    x_max = max(vol_i.max(), vol_f.max()) * 1.05
    slope = (ret_o[0] - r_f) / vol_o[0]
    ax.plot([0, x_max], [r_f, r_f + slope * x_max], color=INK_SECONDARY, lw=1, zorder=3,
            label="Capital Market Line")
    ax.scatter(0, r_f, color=INK, s=30, zorder=4, clip_on=False)
    labels = [ax.annotate("Risk-free", (0, r_f), xytext=(6, -12), textcoords="offset points",
                          fontsize=9, color=INK, path_effects=HALO)]

    markers = [(0, r_f)]
    for name, (w, marker, offset) in portfolios.items():
        v, r = stats(w)
        markers.append((v[0], r[0]))
        ax.scatter(v, r, marker=marker, s=160 if marker == "*" else 70, color=INK,
                   edgecolor=SURFACE, lw=1, zorder=5)
        labels.append(ax.annotate(name, (v[0], r[0]), xytext=offset, textcoords="offset points",
                                  fontsize=9, color=INK, fontweight="bold", ha="right", va="center",
                                  path_effects=HALO, zorder=6))

    ax.set_xlim(0, x_max)
    ax.set_ylim(min(0, ret_f.min(), ret_i.min()), max(ret_i.max(), ret_f.max()) * 1.1)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Annualized volatility", color=INK_SECONDARY)
    ax.set_ylabel("Annualized expected return", color=INK_SECONDARY)
    ax.set_title(f"Long-only efficient frontier, 49 industries, {lookback}-year lookback "
                 f"(estimated at {t:%d %b %Y})", fontsize=12)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK_SECONDARY)
    fig.tight_layout()

    order = np.argsort(sharpe_i)
    chosen = np.concatenate([order[::-1][: n_labels // 2], order[: n_labels // 2]])
    renderer = fig.canvas.get_renderer()
    taken = [legend.get_window_extent(renderer)] + [lab.get_window_extent(renderer) for lab in labels]
    avoid = ax.transData.transform(np.vstack([np.column_stack([vol_i, ret_i]), markers]))
    _place_labels(ax, np.column_stack([vol_i, ret_i])[chosen], np.array(daily.columns)[chosen],
                  sc.cmap(sc.norm(sharpe_i[chosen])), avoid, taken)

    if path:
        fig.savefig(path, dpi=200)
    return fig


def plot_performance_table(performance, period, significance=0.05, path=None):
    """Performance table as a figure, grouped by strategy family.
    Alphas significant at the `significance` level are shown in bold."""
    n = len(performance)
    fig, ax = plt.subplots(figsize=(10, 0.38 * n + 1.5))
    fig.set_facecolor(SURFACE)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(n + 0.6, -0.9)

    x_values = np.linspace(0.36, 0.985, len(TABLE_COLUMNS))
    for x, (header, _) in zip(x_values, TABLE_COLUMNS.values()):
        ax.text(x, 0, header, ha="right", va="center", fontsize=10, color=INK_SECONDARY, fontweight="bold")
    ax.text(0.01, 0, "Strategy", ha="left", va="center", fontsize=10, color=INK_SECONDARY, fontweight="bold")
    ax.axhline(0.6, color=AXIS, lw=1)

    groups = [name.split()[0] if name not in BENCHMARKS else "Benchmark" for name in performance.index]
    for i, (name, row) in enumerate(performance.iterrows(), start=1):
        if i > 1 and groups[i - 1] != groups[i - 2]:
            ax.axhline(i - 0.5, color=GRID, lw=0.8)
        ax.text(0.01, i, _display_name(name), ha="left", va="center", fontsize=10, color=INK)
        significant = row["Alpha p-value"] < significance
        for x, (column, (_, fmt)) in zip(x_values, TABLE_COLUMNS.items()):
            value = row[column]
            text = "–" if pd.isna(row["Alpha p-value"]) and column.startswith("Alpha") else fmt.format(value)
            bold = column == "Alpha" and significant
            ax.text(x, i, text, ha="right", va="center", fontsize=10, color=INK,
                    fontweight="bold" if bold else "normal")
    ax.axhline(n + 0.5, color=AXIS, lw=1)

    ax.set_title(f"Performance, {period}", fontsize=12, color=INK)
    fig.text(0.5, 0.02, "Monthly returns. Sharpe ratio over the 1-month T-bill. Alpha and beta from an OLS CAPM "
             f"regression on the CRSP market. Bold: alpha significant at the {significance:.0%} level.",
             ha="center", fontsize=8, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    if path:
        fig.savefig(path, dpi=200)
    return fig

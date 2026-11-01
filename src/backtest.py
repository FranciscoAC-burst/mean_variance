import numpy as np
import pandas as pd

from .data import EVAL_START
from .optimization import min_variance, optimal_portfolio

LOOKBACKS = (2, 3, 5, 10)


def run_backtest(daily, monthly, market, rf, rebalance_dates, lookbacks=LOOKBACKS, eval_start=EVAL_START):
    """Monthly-rebalanced backtest. Weights are set at the close of the last trading day of
    month m-1 and held (buy-and-hold) during month m.
    Returns monthly strategy returns and, for each strategy, its weights (industries + T-bills)."""
    excess = monthly.sub(rf, axis=0)
    months = monthly.loc[eval_start:].index
    n = monthly.shape[1]
    t_bills = np.append(np.zeros(n), 1.0)

    rows = {f"MinVar {L}y": [] for L in lookbacks} | {f"Optimal {L}y": [] for L in lookbacks} | {"1/N": []}

    for m in months:
        prev = m - 1
        t = rebalance_dates[prev]
        for L in lookbacks:
            window = daily.loc[t - pd.DateOffset(years=L) + pd.Timedelta(days=1): t].dropna()
            mu_window = excess.loc[prev - 12 * L + 1: prev]
            if len(mu_window) < 12 * L:
                raise ValueError(f"Not enough data for a {L}y lookback in {m}")

            cov = window.cov().values
            rows[f"MinVar {L}y"].append(np.append(min_variance(cov), 0.0))

            w = optimal_portfolio(cov, mu_window.mean().values)
            rows[f"Optimal {L}y"].append(t_bills if w is None else np.append(w, 0.0))

        rows["1/N"].append(np.append(np.full(n, 1 / n), 0.0))

    columns = list(monthly.columns) + ["T-bills"]
    weights = {name: pd.DataFrame(np.vstack(r), index=months, columns=columns) for name, r in rows.items()}

    assets = monthly.loc[months].assign(**{"T-bills": rf.loc[months]})
    returns = pd.DataFrame({name: (W * assets).sum(axis=1) for name, W in weights.items()})
    returns["Market"] = market.loc[months]
    return returns, weights

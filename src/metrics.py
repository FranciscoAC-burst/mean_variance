import numpy as np
import pandas as pd
import statsmodels.api as sm


def wealth(returns):
    """Cumulative wealth of $1 invested at the start of the first month."""
    return (1 + returns).cumprod()


def drawdown(returns):
    """Drop from the running peak of wealth (the initial $1 counts as a peak)."""
    w = wealth(returns)
    return w / w.cummax().clip(lower=1) - 1


def performance_table(returns, rf, market):
    """Annualized performance metrics from monthly returns. CAPM alpha/beta by plain OLS."""
    rf = rf.loc[returns.index]
    market_excess = sm.add_constant((market.loc[returns.index] - rf).values)
    years = len(returns) / 12

    rows = {}
    for name, r in returns.items():
        excess = r - rf
        if name == "Market":
            alpha, p_value, beta = 0.0, np.nan, 1.0
        else:
            ols = sm.OLS(excess.values, market_excess).fit()
            alpha, p_value, beta = ols.params[0] * 12, ols.pvalues[0], ols.params[1]
        rows[name] = {
            "CAGR": (1 + r).prod() ** (1 / years) - 1,
            "Volatility": r.std() * np.sqrt(12),
            "Sharpe": excess.mean() / excess.std() * np.sqrt(12),
            "Max drawdown": drawdown(r).min(),
            "Alpha": alpha,
            "Alpha p-value": p_value,
            "Beta": beta,
        }
    return pd.DataFrame(rows).T

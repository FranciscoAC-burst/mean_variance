# Mean-variance portfolios on the 49 Fama-French industries

Out-of-sample test of long-only mean-variance portfolios: does optimizing on historical estimates
beat simple diversification and the market?

## Data

All files come from the [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
(CRSP database 202609) and are stored, unmodified, in `data/`.

| File | Frequency | Used for |
|---|---|---|
| `49_Industry_Portfolios_daily_CSV.zip` | Daily | Covariance matrix Σ |
| `49_Industry_Portfolios_CSV.zip` | Monthly | Expected returns μ (optimal portfolio) and holding-period returns |
| `F-F_Research_Data_Factors_CSV.zip` | Monthly | CRSP market (Mkt-RF + RF) and risk-free rate |

Only value-weighted industry returns are used. `Siccodes49.zip` (industry definitions) is included for
reference and is not used by the code.

## Methodology

- **Strategies**, rebalanced monthly:
  - Minimum variance, long-only, with 2, 3, 5 and 10-year lookbacks.
  - Optimal portfolio, long-only, with 2, 3, 5 and 10-year lookbacks: the portfolio with the highest
    estimated Sharpe ratio (the tangency portfolio of mean-variance theory). If no industry has a
    positive expected excess return, it holds 100% T-bills.
  - Equal weight (1/N) across the 49 industries.
  - Benchmark: CRSP total market.
- **Estimation**: sample covariance of daily returns over the last L years; expected returns are the
  mean of the last L × 12 monthly excess returns.
- **Timing**: weights are computed with data up to the close of the last trading day of each month and
  held (buy-and-hold) over the following month.
- **Sample**: data from January 1970; performance is evaluated from January 1980 (when the 10-year
  lookback is first available) to September 2026.
- **Metrics**: CAGR, annualized volatility, Sharpe ratio, maximum drawdown, CAPM alpha (with its OLS
  p-value) and beta, all from monthly returns.

## Repository structure

```
mean_variance/
├── data/                  # French Data Library zips
├── src/
│   ├── data.py            # reads the zips into return series
│   ├── optimization.py    # minimum variance, optimal portfolio and efficient frontier
│   ├── backtest.py        # monthly rebalancing loop
│   ├── metrics.py         # performance metrics
│   └── plots.py           # figures
├── main.ipynb             # runs everything and writes output/
├── output/
│   ├── tables/            # performance.csv, monthly_returns.csv
│   └── figures/           # wealth and efficient frontier charts
└── LICENSE
```

## How to run

Requires Python 3.10+ with `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib` and `jupyter`.

Run all cells of `main.ipynb` from the repository root. The backtest takes about 70 seconds; tables and
figures are written to `output/`.

## Limitations

- **One-day look-ahead**: weights use the closing prices of the same day the portfolio is rebalanced.
- **Industry classification**: the 49-industry scheme was designed in 1997, after part of the sample.
- **Data revisions**: French's data are revised over time as CRSP is updated.
- **No transaction costs**.

## License

Code licensed under the [MIT License](LICENSE). The data files in `data/` are © Eugene F. Fama and
Kenneth R. French and are not covered by this license.

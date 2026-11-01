import numpy as np
from scipy.optimize import minimize

SLSQP_OPTIONS = {"ftol": 1e-12, "maxiter": 1000}


def _solve_qp(cov, x0, bounds, constraints):
    """min x' cov x via SLSQP. cov is rescaled for numerical conditioning (argmin unchanged)."""
    cov = cov / np.mean(np.diag(cov))
    res = minimize(
        lambda x: x @ cov @ x,
        x0,
        jac=lambda x: 2 * cov @ x,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=SLSQP_OPTIONS,
    )
    if not res.success:
        raise RuntimeError(f"Optimizer did not converge: {res.message}")
    x = np.clip(res.x, 0, None)
    return x / x.sum()


def _budget(n):
    return {"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}


def min_variance(cov):
    """Long-only minimum variance: min w'Σw  s.t.  sum(w) = 1, w >= 0."""
    n = len(cov)
    return _solve_qp(cov, np.full(n, 1 / n), [(0, 1)] * n, [_budget(n)])


def efficient_frontier(cov, mu, n_points=100):
    """Long-only minimum-variance frontier between the lowest and highest expected return:
    for each target, min w'Σw  s.t.  μ'w = target, sum(w) = 1, w >= 0. Returns (targets, weights)."""
    n = len(cov)
    targets = np.linspace(mu.min(), mu.max(), n_points)
    weights = []
    for target in targets:
        on_target = {"type": "eq", "fun": lambda w, t=target: mu @ w - t, "jac": lambda w: mu}
        weights.append(_solve_qp(cov, np.full(n, 1 / n), [(0, 1)] * n, [_budget(n), on_target]))
    return targets, np.array(weights)


def optimal_portfolio(cov, mu):
    """Long-only portfolio with the highest Sharpe ratio, solved as  min y'Σy  s.t.  μ'y = 1, y >= 0;
    w = y / sum(y).
    `mu` are expected excess returns. Returns None if no asset has positive expected excess return."""
    if mu.max() <= 0:
        return None
    mu = mu / mu.max()
    n = len(cov)
    target = {"type": "eq", "fun": lambda y: mu @ y - 1, "jac": lambda y: mu}
    return _solve_qp(cov, (mu == 1).astype(float), [(0, None)] * n, [target])

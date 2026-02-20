from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class PortfolioOptimizationResult:
    weights: dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe: float


class ConvexPortfolioOptimizer:
    def __init__(self, risk_free_rate: float = 0.0) -> None:
        self.risk_free_rate = risk_free_rate

    @staticmethod
    def _sanitize_returns(returns: pd.DataFrame) -> pd.DataFrame:
        clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        if clean.empty:
            raise ValueError("Returns vacíos o corruptos para optimización")
        return clean

    def mean_variance(
        self,
        returns: pd.DataFrame,
        target_return: float,
        exposure_limit: float = 1.0,
    ) -> PortfolioOptimizationResult:
        ret = self._sanitize_returns(returns)
        mu = ret.mean().to_numpy(dtype=float)
        sigma = ret.cov().to_numpy(dtype=float)
        n = len(ret.columns)

        try:
            import cvxpy as cp

            w = cp.Variable(n)
            objective = cp.Minimize(cp.quad_form(w, sigma + np.eye(n) * 1e-8))
            constraints = [
                mu @ w >= target_return,
                cp.norm1(w) <= exposure_limit,
                w >= 0,
                cp.sum(w) == 1,
            ]
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS, verbose=False)
            if w.value is None:
                raise RuntimeError("Mean-variance no convergió")
            weights = np.clip(np.asarray(w.value, dtype=float), 0.0, 1.0)
        except Exception:
            inv_diag = 1.0 / np.clip(np.diag(sigma), 1e-9, None)
            positive_mu = np.clip(mu, 0.0, None)
            weights = positive_mu * inv_diag
            if weights.sum() <= 0:
                weights = inv_diag
        weights = weights / max(weights.sum(), 1e-9)
        return self._build_result(ret.columns.tolist(), weights, mu, sigma)

    def risk_parity(self, returns: pd.DataFrame, iterations: int = 200) -> PortfolioOptimizationResult:
        ret = self._sanitize_returns(returns)
        sigma = ret.cov().to_numpy(dtype=float) + np.eye(len(ret.columns)) * 1e-9
        n = len(ret.columns)
        w = np.full(n, 1.0 / n)
        target = 1.0 / n

        for _ in range(iterations):
            port_var = float(w.T @ sigma @ w)
            mrc = sigma @ w
            rc = w * mrc / max(port_var, 1e-12)
            grad = rc - target
            w = np.clip(w - 0.05 * grad, 1e-6, 1.0)
            w = w / w.sum()

        mu = ret.mean().to_numpy(dtype=float)
        return self._build_result(ret.columns.tolist(), w, mu, sigma)

    def kelly_constrained(self, returns: pd.DataFrame, max_drawdown: float = 0.20) -> PortfolioOptimizationResult:
        ret = self._sanitize_returns(returns)
        mu = ret.mean().to_numpy(dtype=float)
        sigma = ret.cov().to_numpy(dtype=float) + np.eye(len(ret.columns)) * 1e-8
        raw = np.linalg.solve(sigma, mu)
        raw = np.clip(raw, 0.0, None)
        if raw.sum() <= 0:
            raw = np.ones_like(raw)
        raw /= raw.sum()

        scale = float(np.clip((1.0 - max_drawdown) / 0.8, 0.05, 1.0))
        weights = raw * scale
        weights /= max(weights.sum(), 1e-9)
        return self._build_result(ret.columns.tolist(), weights, mu, sigma)

    def correlation_aware_allocation(self, returns: pd.DataFrame, corr_threshold: float = 0.80) -> PortfolioOptimizationResult:
        ret = self._sanitize_returns(returns)
        corr = ret.corr().abs().to_numpy(dtype=float)
        mu = ret.mean().to_numpy(dtype=float)
        sigma = ret.cov().to_numpy(dtype=float)

        penalties = np.ones(len(ret.columns))
        for i in range(len(ret.columns)):
            high_corr = corr[i] > corr_threshold
            penalties[i] = 1.0 / (1.0 + float(np.sum(high_corr)) * 0.25)

        weights = np.clip(mu, 0.0, None) * penalties
        if weights.sum() <= 0:
            weights = penalties
        weights = weights / max(weights.sum(), 1e-9)
        return self._build_result(ret.columns.tolist(), weights, mu, sigma)

    def _build_result(self, names: list[str], weights: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> PortfolioOptimizationResult:
        expected_return = float(weights @ mu)
        expected_vol = float(np.sqrt(max(weights.T @ sigma @ weights, 1e-12)))
        sharpe = float((expected_return - self.risk_free_rate) / max(expected_vol, 1e-12))
        return PortfolioOptimizationResult(
            weights={name: float(w) for name, w in zip(names, weights)},
            expected_return=expected_return,
            expected_volatility=expected_vol,
            sharpe=sharpe,
        )

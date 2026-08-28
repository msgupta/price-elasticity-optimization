"""
mllib.py -- lightweight, dependency-free (numpy-only) ML primitives.

Built from first principles because the target deployment environments for
these projects (retail edge servers / lightweight batch jobs) did not always
have scikit-learn available, and because owning the math end-to-end made it
easy to reason about production failure modes (e.g. what happens to a CART
split when a feature has 90% zeros, or how Ridge regularization interacts
with collinear lag features).

Implements:
    - RidgeRegression            : closed-form L2-regularized linear regression
    - DecisionTreeRegressor      : CART, variance-reduction splits
    - GradientBoostingRegressor  : additive boosting on top of shallow CART trees
    - HoltWinters                : triple exponential smoothing (additive/multiplicative)
    - metrics                    : MAPE, WAPE, RMSE, MAE, R2

Author: Mani Shankr Gupta
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def mape(y_true, y_pred, eps=1e-6):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > eps
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def wape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100) if denom else float("nan")


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot else float("nan")


def regression_report(y_true, y_pred) -> dict:
    return {
        "MAPE_%": round(mape(y_true, y_pred), 2),
        "WAPE_%": round(wape(y_true, y_pred), 2),
        "RMSE": round(rmse(y_true, y_pred), 3),
        "MAE": round(mae(y_true, y_pred), 3),
        "R2": round(r2_score(y_true, y_pred), 4),
        "Accuracy_%_(100-MAPE)": round(100 - mape(y_true, y_pred), 2),
    }


# --------------------------------------------------------------------------- #
# Ridge Regression (closed form)
# --------------------------------------------------------------------------- #
class RidgeRegression:
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
        self._mu = None
        self._sigma = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        self._sigma[self._sigma == 0] = 1.0
        Xs = (X - self._mu) / self._sigma
        n, d = Xs.shape
        if self.fit_intercept:
            Xd = np.hstack([np.ones((n, 1)), Xs])
            reg = self.alpha * np.eye(d + 1)
            reg[0, 0] = 0.0
        else:
            Xd = Xs
            reg = self.alpha * np.eye(d)
        beta = np.linalg.solve(Xd.T @ Xd + reg, Xd.T @ y)
        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xs = (X - self._mu) / self._sigma
        return Xs @ self.coef_ + self.intercept_


# --------------------------------------------------------------------------- #
# CART Decision Tree Regressor
# --------------------------------------------------------------------------- #
class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, value=None):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = value


class DecisionTreeRegressor:
    def __init__(self, max_depth=4, min_samples_split=10, min_samples_leaf=5, n_feature_candidates=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_feature_candidates = n_feature_candidates
        self.root_ = None
        self.feature_importances_ = None
        self._n_features = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._n_features = X.shape[1]
        self._importance_accum = np.zeros(self._n_features)
        self.root_ = self._build(X, y, depth=0)
        total = self._importance_accum.sum()
        self.feature_importances_ = (
            self._importance_accum / total if total > 0 else self._importance_accum
        )
        return self

    def _build(self, X, y, depth):
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or np.all(y == y[0])
        ):
            return _Node(value=float(np.mean(y)))

        best = self._best_split(X, y)
        if best is None:
            return _Node(value=float(np.mean(y)))

        feat, thresh, gain = best
        self._importance_accum[feat] += gain * len(y)
        left_mask = X[:, feat] <= thresh
        node = _Node()
        node.feature, node.threshold = feat, thresh
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _best_split(self, X, y):
        n, d = X.shape
        parent_var = y.var()
        best_gain, best_feat, best_thresh = -1.0, None, None

        feature_idx = range(d)
        if self.n_feature_candidates:
            feature_idx = np.random.choice(d, size=min(self.n_feature_candidates, d), replace=False)

        for feat in feature_idx:
            col = X[:, feat]
            candidates = np.unique(np.quantile(col, np.linspace(0.05, 0.95, 10)))
            for thresh in candidates:
                left = y[col <= thresh]
                right = y[col > thresh]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue
                weighted_var = (len(left) * left.var() + len(right) * right.var()) / n
                gain = parent_var - weighted_var
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh

        if best_feat is None:
            return None
        return best_feat, best_thresh, best_gain

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row, self.root_) for row in X])

    def _predict_one(self, row, node):
        while node.value is None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value


# --------------------------------------------------------------------------- #
# Gradient Boosting Regressor (additive CART ensemble)
# --------------------------------------------------------------------------- #
class GradientBoostingRegressor:
    def __init__(self, n_estimators=150, learning_rate=0.08, max_depth=3,
                 min_samples_leaf=5, subsample=0.8, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        self.trees_ = []
        self.init_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.init_ = float(np.mean(y))
        pred = np.full(n, self.init_)
        importances = np.zeros(d)

        for _ in range(self.n_estimators):
            residual = y - pred
            if self.subsample < 1.0:
                idx = rng.choice(n, size=int(n * self.subsample), replace=False)
            else:
                idx = np.arange(n)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_leaf * 2,
            )
            tree.fit(X[idx], residual[idx])
            update = tree.predict(X)
            pred += self.learning_rate * update
            self.trees_.append(tree)
            importances += tree.feature_importances_

        self.feature_importances_ = importances / importances.sum() if importances.sum() else importances
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.init_)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(X)
        return pred


# --------------------------------------------------------------------------- #
# Holt-Winters triple exponential smoothing
# --------------------------------------------------------------------------- #
class HoltWinters:
    def __init__(self, season_length=7, alpha=0.3, beta=0.1, gamma=0.2, trend="add", seasonal="add", phi=0.97):
        self.season_length = season_length
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.trend, self.seasonal = trend, seasonal
        self.phi = phi  # damped-trend factor: prevents runaway extrapolation over long horizons
        self.level_, self.trend_, self.season_ = None, None, None

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        L = self.season_length
        n = len(y)
        season = np.array([y[i::L].mean() for i in range(L)])
        season = season - season.mean() if self.seasonal == "add" else season / season.mean()
        level = y[:L].mean()
        trend = (y[L:2 * L].mean() - y[:L].mean()) / L

        levels, trends, seasons = [level], [trend], list(season)
        fitted = []
        for t in range(n):
            s = seasons[t % L]
            if self.seasonal == "add":
                yhat = level + trend + s
            else:
                yhat = (level + trend) * s
            fitted.append(yhat)

            if self.seasonal == "add":
                new_level = self.alpha * (y[t] - s) + (1 - self.alpha) * (level + trend)
            else:
                new_level = self.alpha * (y[t] / s) + (1 - self.alpha) * (level + trend)
            new_trend = self.beta * (new_level - level) + (1 - self.beta) * trend
            if self.seasonal == "add":
                new_season = self.gamma * (y[t] - new_level) + (1 - self.gamma) * s
            else:
                new_season = self.gamma * (y[t] / new_level) + (1 - self.gamma) * s

            level, trend = new_level, new_trend
            seasons[t % L] = new_season
            levels.append(level)
            trends.append(trend)

        self.level_, self.trend_, self.season_ = level, trend, seasons
        self.fitted_ = np.array(fitted)
        return self

    def forecast(self, steps):
        preds = []
        damped_trend_sum = 0.0
        phi_pow = 1.0
        for h in range(1, steps + 1):
            phi_pow *= self.phi
            damped_trend_sum += phi_pow
            s = self.season_[(h - 1) % self.season_length]
            if self.seasonal == "add":
                preds.append(self.level_ + damped_trend_sum * self.trend_ + s)
            else:
                preds.append((self.level_ + damped_trend_sum * self.trend_) * s)
        return np.array(preds)

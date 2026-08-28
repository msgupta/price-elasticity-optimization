"""
Estimates own-price elasticity per SKU via a two-stage hierarchical (empirical-
Bayes shrinkage) log-log demand model:

  Stage 1 (category level): Ridge log-log regression estimates the *average*
      own-price elasticity, cross-price elasticity, promo lift, and seasonality
      for each category. Pooling across ~1,800 SKUs per category makes this
      estimate stable.

  Stage 2 (SKU level): for each SKU, partial out the category-level cross-price/
      promo/seasonal effects, then run a simple OLS of the residual demand on
      log(price) using only that SKU's ~26 weekly observations. This raw
      per-SKU estimate is noisy (26 points, ~10% price variation), so it is
      shrunk toward the category mean via a James-Stein / empirical-Bayes
      estimator: SKUs with more precise individual signal (low standard error)
      keep more of their own estimate; SKUs with noisy individual signal are
      pulled toward the category average.

This mirrors the real production approach: a single price-response coefficient
per category is too coarse for pricing decisions at the SKU level, but a fully
independent per-SKU fit overfits on 26 weekly points. Partial pooling gets
useful SKU-level differentiation without the overfitting.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mllib import RidgeRegression, regression_report

ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLS = ["log_price", "log_competitor_price", "promo", "week_sin", "week_cos"]


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel[panel["units_sold"] > 0].copy()
    df["log_units"] = np.log(df["units_sold"])
    df["log_price"] = np.log(df["price"])
    df["log_competitor_price"] = np.log(df["competitor_price"])
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 26)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 26)
    return df


def fit_category_models(df: pd.DataFrame):
    category_models, category_elasticity = {}, {}
    for cat, sub in df.groupby("category"):
        X, y = sub[FEATURE_COLS].values, sub["log_units"].values
        model = RidgeRegression(alpha=2.0).fit(X, y)
        category_models[cat] = model
        idx_price = FEATURE_COLS.index("log_price")
        category_elasticity[cat] = model.coef_[idx_price] / model._sigma[idx_price]
    return category_models, category_elasticity


def _sku_ols_elasticity(sub, cat_model, cat_elasticity):
    """Partial out category-level cross-price/promo/season effect, then OLS the residual on log_price."""
    idx = {c: i for i, c in enumerate(FEATURE_COLS)}
    other_cols = [c for c in FEATURE_COLS if c != "log_price"]

    other_effect = np.zeros(len(sub))
    for c in other_cols:
        raw_coef = cat_model.coef_[idx[c]] / cat_model._sigma[idx[c]]
        other_effect += raw_coef * (sub[c].values - cat_model._mu[idx[c]])
    intercept_adj = cat_model.intercept_ - np.sum(
        [cat_model.coef_[idx[c]] / cat_model._sigma[idx[c]] * cat_model._mu[idx[c]] for c in FEATURE_COLS]
    )
    adjusted_y = sub["log_units"].values - other_effect - intercept_adj

    x = sub["log_price"].values
    n = len(x)
    x_mean, y_mean = x.mean(), adjusted_y.mean()
    sxx = np.sum((x - x_mean) ** 2)
    if sxx < 1e-8 or n < 5:
        return cat_elasticity, 999.0
    beta = np.sum((x - x_mean) * (adjusted_y - y_mean)) / sxx
    resid = adjusted_y - (y_mean + beta * (x - x_mean))
    sigma2 = np.sum(resid ** 2) / max(n - 2, 1)
    se_beta = float(np.sqrt(sigma2 / sxx))
    return float(beta), se_beta


def fit_sku_elasticity_shrinkage(df: pd.DataFrame, category_models, category_elasticity):
    rows = []
    for cat, sub_cat in df.groupby("category"):
        cat_model = category_models[cat]
        raw_estimates, ses, sku_ids = [], [], []
        for sku_id, sub in sub_cat.groupby("sku_id"):
            e_raw, se = _sku_ols_elasticity(sub, cat_model, category_elasticity[cat])
            raw_estimates.append(e_raw); ses.append(se); sku_ids.append(sku_id)

        raw_estimates = np.array(raw_estimates)
        ses = np.array(ses)
        valid = ses < 900
        between_var = max(np.var(raw_estimates[valid]) - np.mean(ses[valid] ** 2), 1e-4)

        for sku_id, e_raw, se in zip(sku_ids, raw_estimates, ses):
            if se >= 900:
                shrunk = category_elasticity[cat]
            else:
                w = between_var / (between_var + se ** 2)
                shrunk = w * e_raw + (1 - w) * category_elasticity[cat]
            rows.append({"sku_id": sku_id, "category": cat, "predicted_elasticity": shrunk, "raw_elasticity": e_raw})

    return pd.DataFrame(rows)


def evaluate_against_ground_truth(sku_elasticity: pd.DataFrame):
    truth = pd.read_csv(ROOT / "data" / "raw" / "true_elasticity_holdout.csv")
    merged = sku_elasticity.merge(truth, on=["sku_id", "category"])

    cat_pred = merged.groupby("category")["predicted_elasticity"].mean()
    cat_true = merged.groupby("category")["true_elasticity"].mean()
    rank_corr = float(np.corrcoef(cat_pred.rank(), cat_true.rank())[0, 1])

    merged["pred_above_median"] = merged.groupby("category")["predicted_elasticity"].transform(lambda s: s > s.median())
    merged["true_above_median"] = merged.groupby("category")["true_elasticity"].transform(lambda s: s > s.median())
    directional_accuracy = float((merged["pred_above_median"] == merged["true_above_median"]).mean() * 100)

    magnitude_report = regression_report(merged["true_elasticity"], merged["predicted_elasticity"])
    return merged, {
        "category_rank_correlation": round(rank_corr, 3),
        "sku_directional_accuracy_%": round(directional_accuracy, 2),
        **magnitude_report,
    }


if __name__ == "__main__":
    panel = pd.read_csv(ROOT / "data" / "raw" / "sku_price_demand_panel.csv")
    df = build_features(panel)
    category_models, category_elasticity = fit_category_models(df)
    sku_elasticity = fit_sku_elasticity_shrinkage(df, category_models, category_elasticity)
    merged, metrics = evaluate_against_ground_truth(sku_elasticity)
    print(metrics)

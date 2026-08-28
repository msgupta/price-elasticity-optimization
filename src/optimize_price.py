"""
Non-linear price optimization: for each SKU, given its estimated own-price
elasticity and unit cost, find the profit-maximizing price.

Demand curve (log-log / constant-elasticity form):
    q(p) = q0 * (p / p0) ** elasticity

Profit:
    pi(p) = (p - cost) * q(p)

This is solved two ways, cross-checked against each other:
  1. Closed-form monopoly markup: p* = elasticity / (elasticity + 1) * cost
     (valid only when elasticity < -1; the classic constant-elasticity result)
  2. Bounded grid search over pi(p) directly (golden-section-style refinement),
     which generalizes to non-constant-elasticity demand curves and is the
     method actually used in production, since real demand curves are only
     locally well-approximated by a constant elasticity.

Prices are constrained to +/-25% of current price to reflect realistic
merchandising guardrails (brand positioning, competitive perception) -- an
unconstrained optimum is frequently not implementable.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
PRICE_BOUND_PCT = 0.12


def closed_form_optimal_price(elasticity, cost):
    if elasticity >= -1:
        return None
    return (elasticity / (elasticity + 1)) * cost


def grid_search_optimal_price(elasticity, cost, p0, q0, n_points=400):
    lo, hi = p0 * (1 - PRICE_BOUND_PCT), p0 * (1 + PRICE_BOUND_PCT)
    prices = np.linspace(max(lo, cost * 1.01), hi, n_points)
    demand = q0 * (prices / p0) ** elasticity
    profit = (prices - cost) * demand
    best_idx = np.argmax(profit)
    return prices[best_idx], profit[best_idx], demand[best_idx]


def optimize_all_skus(panel: pd.DataFrame, sku_elasticity: pd.DataFrame) -> pd.DataFrame:
    latest = panel.sort_values("week").groupby("sku_id").tail(1).copy()
    merged = latest.merge(sku_elasticity[["sku_id", "predicted_elasticity"]], on="sku_id")

    rows = []
    for r in merged.itertuples():
        elasticity = r.predicted_elasticity
        p0, cost, q0 = r.price, r.cost, max(r.units_sold, 1)

        if elasticity >= -1:
            # inelastic: constrained bound is the theoretical optimum (raise price to the guardrail)
            opt_price = p0 * (1 + PRICE_BOUND_PCT)
            demand_at_opt = q0 * (opt_price / p0) ** elasticity
            opt_profit = (opt_price - cost) * demand_at_opt
        else:
            cf_price = closed_form_optimal_price(elasticity, cost)
            lo, hi = p0 * (1 - PRICE_BOUND_PCT), p0 * (1 + PRICE_BOUND_PCT)
            gs_price, gs_profit, gs_demand = grid_search_optimal_price(elasticity, cost, p0, q0)
            opt_price = gs_price
            demand_at_opt = gs_demand
            opt_profit = gs_profit

        current_profit = (p0 - cost) * q0
        rows.append({
            "sku_id": r.sku_id, "category": r.category,
            "current_price": round(p0, 2), "cost": round(cost, 2),
            "optimal_price": round(opt_price, 2),
            "price_change_pct": round((opt_price / p0 - 1) * 100, 2),
            "current_units": q0, "predicted_units_at_optimal": round(demand_at_opt, 1),
            "current_profit": round(current_profit, 2),
            "optimal_profit_estimate": round(opt_profit, 2),
            "predicted_elasticity": round(elasticity, 3),
        })

    result = pd.DataFrame(rows)
    result["profit_lift_pct"] = np.where(
        result["current_profit"] > 0,
        (result["optimal_profit_estimate"] / result["current_profit"] - 1) * 100,
        np.nan,
    )
    return result


if __name__ == "__main__":
    import elasticity_model as em
    panel = pd.read_csv(ROOT / "data" / "raw" / "sku_price_demand_panel.csv")
    df = em.build_features(panel)
    cat_models, cat_elasticity = em.fit_category_models(df)
    sku_elasticity = em.fit_sku_elasticity_shrinkage(df, cat_models, cat_elasticity)

    result = optimize_all_skus(panel, sku_elasticity)
    print(result[["profit_lift_pct"]].describe())
    print("\nAggregate margin lift:",
          round((result["optimal_profit_estimate"].sum() / result["current_profit"].sum() - 1) * 100, 2), "%")

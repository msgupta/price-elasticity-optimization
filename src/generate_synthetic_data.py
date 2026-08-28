"""
Generates a synthetic SKU-level weekly price/demand panel across 10,000+ SKUs
spanning multiple categories, each with a distinct ground-truth price
elasticity, competitive cross-elasticity, and promo lift.

Ground-truth elasticities are saved separately (data/raw/true_elasticity.csv)
purely for offline validation of the estimation model further down the
pipeline -- they are never used as a model input.
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

CATEGORIES = {
    "Apparel":        {"elasticity": -1.8, "cross_elasticity": 0.5, "cost_margin": 0.42},
    "Home & Decor":   {"elasticity": -1.3, "cross_elasticity": 0.3, "cost_margin": 0.48},
    "Footwear":       {"elasticity": -2.1, "cross_elasticity": 0.6, "cost_margin": 0.40},
    "Beauty":         {"elasticity": -1.1, "cross_elasticity": 0.2, "cost_margin": 0.55},
    "Electronics":    {"elasticity": -2.6, "cross_elasticity": 0.7, "cost_margin": 0.28},
    "Jewelry":        {"elasticity": -0.9, "cross_elasticity": 0.15, "cost_margin": 0.60},
}


def generate(n_skus_per_cat=1800, n_weeks=26, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    true_elasticity_rows = []

    sku_id = 0
    for cat, params in CATEGORIES.items():
        base_elasticity = params["elasticity"]
        for _ in range(n_skus_per_cat):
            sku_id += 1
            sku = f"SKU-{sku_id:06d}"
            sku_elasticity = base_elasticity + rng.normal(0, 0.18)
            cross_elasticity = params["cross_elasticity"] + rng.normal(0, 0.05)
            base_price = rng.uniform(12, 180)
            cost = base_price * (1 - params["cost_margin"]) * rng.uniform(0.9, 1.1)
            base_demand = rng.uniform(80, 900)

            true_elasticity_rows.append({"sku_id": sku, "category": cat, "true_elasticity": sku_elasticity})

            for week in range(n_weeks):
                price_shock = rng.normal(0, 0.10)
                price = base_price * np.exp(price_shock)
                competitor_price = base_price * np.exp(rng.normal(0.01, 0.08))
                promo = 1 if rng.random() < 0.12 else 0
                season = 1 + 0.15 * np.sin(2 * np.pi * week / 26)

                log_demand = (
                    np.log(base_demand)
                    + sku_elasticity * np.log(price / base_price)
                    + cross_elasticity * np.log(competitor_price / base_price)
                    + 0.35 * promo
                    + np.log(season)
                    + rng.normal(0, 0.12)
                )
                units_sold = max(0, np.random.default_rng(seed + sku_id * 100 + week).poisson(np.exp(log_demand)))

                rows.append({
                    "sku_id": sku, "category": cat, "week": week,
                    "price": round(price, 2), "competitor_price": round(competitor_price, 2),
                    "cost": round(cost, 2), "promo": promo, "units_sold": units_sold,
                })

    panel = pd.DataFrame(rows)
    true_elasticity = pd.DataFrame(true_elasticity_rows)
    return panel, true_elasticity


if __name__ == "__main__":
    panel, true_elasticity = generate()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_DIR / "sku_price_demand_panel.csv", index=False)
    true_elasticity.to_csv(OUT_DIR / "true_elasticity_holdout.csv", index=False)
    print(f"Panel: {len(panel):,} rows across {panel['sku_id'].nunique():,} SKUs")
    print(f"True elasticity holdout: {len(true_elasticity):,} SKUs")

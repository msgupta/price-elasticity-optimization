"""Translates the aggregate margin lift from price optimization into a dollar estimate."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ASSUMPTIONS = {
    "annual_priced_category_revenue_usd": 180_000_000,
    "baseline_gross_margin_pct": 42.0,
    "rollout_coverage_pct": 0.65,   # share of SKUs where the recommended price is actually adopted
}


def estimate_annual_uplift():
    metrics = json.load(open(ROOT / "reports" / "metrics.json"))
    margin_lift_pct = metrics["aggregate_margin_lift_pct"] / 100

    a = ASSUMPTIONS
    baseline_annual_margin = a["annual_priced_category_revenue_usd"] * (a["baseline_gross_margin_pct"] / 100)
    theoretical_uplift = baseline_annual_margin * margin_lift_pct
    realized_uplift = theoretical_uplift * a["rollout_coverage_pct"]

    return {
        "elasticity_model_accuracy_pct": metrics["elasticity_model"]["Accuracy_%_(100-MAPE)"],
        "n_skus_priced": metrics["n_skus_priced"],
        "aggregate_margin_lift_pct": metrics["aggregate_margin_lift_pct"],
        "baseline_annual_gross_margin_usd": round(baseline_annual_margin, 0),
        "theoretical_annual_margin_uplift_usd": round(theoretical_uplift, 0),
        "realized_annual_margin_uplift_usd_at_65pct_rollout": round(realized_uplift, 0),
    }


if __name__ == "__main__":
    result = estimate_annual_uplift()
    print(json.dumps(result, indent=2))
    with open(ROOT / "reports" / "business_impact.json", "w") as f:
        json.dump({"assumptions": ASSUMPTIONS, "result": result}, f, indent=2)

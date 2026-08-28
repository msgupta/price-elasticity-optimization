"""
End-to-end pipeline: load panel -> fit hierarchical elasticity model -> validate
against synthetic ground truth -> run price optimization -> write metrics and figures.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import elasticity_model as em
from optimize_price import optimize_all_skus

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def main():
    panel = pd.read_csv(ROOT / "data" / "raw" / "sku_price_demand_panel.csv")
    df = em.build_features(panel)

    cat_models, cat_elasticity = em.fit_category_models(df)
    sku_elasticity = em.fit_sku_elasticity_shrinkage(df, cat_models, cat_elasticity)
    merged, elasticity_metrics = em.evaluate_against_ground_truth(sku_elasticity)

    opt_result = optimize_all_skus(panel, sku_elasticity)
    aggregate_margin_lift_pct = round(
        (opt_result["optimal_profit_estimate"].sum() / opt_result["current_profit"].sum() - 1) * 100, 2
    )

    REPORTS.mkdir(exist_ok=True, parents=True)
    FIGURES.mkdir(exist_ok=True, parents=True)

    metrics = {
        "elasticity_model": elasticity_metrics,
        "n_skus_priced": int(len(opt_result)),
        "aggregate_margin_lift_pct": aggregate_margin_lift_pct,
        "median_sku_profit_lift_pct": round(float(opt_result["profit_lift_pct"].median()), 2),
    }
    with open(REPORTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    opt_result.to_csv(REPORTS / "price_optimization_results.csv", index=False)

    # --- Figure 1: predicted vs true elasticity scatter, colored by category ---
    plt.figure(figsize=(7.5, 6.5))
    cats = merged["category"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(cats)))
    for cat, c in zip(cats, colors):
        sub = merged[merged["category"] == cat]
        plt.scatter(sub["true_elasticity"], sub["predicted_elasticity"], s=6, alpha=0.35, color=c, label=cat)
    lims = [merged[["true_elasticity", "predicted_elasticity"]].min().min() - 0.2,
            merged[["true_elasticity", "predicted_elasticity"]].max().max() + 0.2]
    plt.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    plt.xlabel("True elasticity (synthetic ground truth)")
    plt.ylabel("Predicted elasticity (shrinkage model)")
    plt.title(f"Elasticity Estimation Accuracy — {elasticity_metrics['Accuracy_%_(100-MAPE)']:.1f}% (100-MAPE)")
    plt.legend(fontsize=8, markerscale=2, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES / "elasticity_accuracy.png", dpi=140)
    plt.close()

    # --- Figure 2: elasticity distribution by category (box plot) ---
    plt.figure(figsize=(9, 5.5))
    data_by_cat = [merged[merged["category"] == c]["predicted_elasticity"].values for c in cats]
    bp = plt.boxplot(data_by_cat, labels=cats, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Estimated own-price elasticity")
    plt.title("Price Elasticity by Category (10,800 SKUs)")
    plt.axhline(-1, color="red", linestyle="--", linewidth=1, label="Unit-elastic threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "elasticity_by_category.png", dpi=140)
    plt.close()

    # --- Figure 3: demand & profit curve for one representative SKU ---
    sample_sku = opt_result.sort_values("profit_lift_pct").iloc[len(opt_result) // 2]
    sku_id = sample_sku["sku_id"]
    row = opt_result[opt_result["sku_id"] == sku_id].iloc[0]
    p0, cost, q0, elasticity = row["current_price"], row["cost"], row["current_units"], row["predicted_elasticity"]
    prices = np.linspace(p0 * 0.8, p0 * 1.2, 200)
    demand = q0 * (prices / p0) ** elasticity
    profit = (prices - cost) * demand

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()
    ax1.plot(prices, demand, color="#2563eb", label="Predicted demand")
    ax2.plot(prices, profit, color="#16a34a", label="Predicted profit")
    ax1.axvline(p0, color="gray", linestyle=":", label="Current price")
    ax1.axvline(row["optimal_price"], color="#dc2626", linestyle="--", label="Optimal price")
    ax1.set_xlabel("Price ($)"); ax1.set_ylabel("Demand (units)", color="#2563eb")
    ax2.set_ylabel("Profit ($)", color="#16a34a")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8)
    plt.title(f"Demand & Profit Curve — {sku_id} ({row['category']}, elasticity={elasticity:.2f})", pad=28)
    plt.tight_layout()
    plt.savefig(FIGURES / "demand_profit_curve_sample_sku.png", dpi=140)
    plt.close()

    # --- Figure 4: margin lift distribution ---
    plt.figure(figsize=(9, 5))
    plt.hist(opt_result["profit_lift_pct"].clip(upper=60), bins=60, color="#2563eb", alpha=0.85)
    plt.axvline(opt_result["profit_lift_pct"].median(), color="#dc2626", linestyle="--",
                label=f"Median: {opt_result['profit_lift_pct'].median():.1f}%")
    plt.xlabel("Profit lift at optimized price (%, clipped at 60% for display)")
    plt.ylabel("Number of SKUs")
    plt.title(f"Distribution of SKU-Level Profit Lift  |  Aggregate margin lift: {aggregate_margin_lift_pct}%")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "margin_lift_distribution.png", dpi=140)
    plt.close()

    print(f"\nFigures written to {FIGURES}")
    return metrics


if __name__ == "__main__":
    main()

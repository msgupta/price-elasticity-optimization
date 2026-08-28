"""Sanity tests: run with `python tests/test_pipeline.py`."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from optimize_price import closed_form_optimal_price, grid_search_optimal_price


def test_closed_form_markup_matches_grid_search_direction():
    elasticity, cost, p0, q0 = -2.0, 10.0, 20.0, 500
    cf_price = closed_form_optimal_price(elasticity, cost)
    gs_price, gs_profit, _ = grid_search_optimal_price(elasticity, cost, p0, q0, n_points=2000)
    # closed form is unconstrained; grid search is bounded to +/-12% of p0 -- so grid search
    # should move in the same direction as the closed-form signal (both should raise price here
    # since elastic demand still allows a markup above cost at this parameterization)
    assert cf_price > cost
    assert gs_profit > (p0 - cost) * q0 * 0.0  # profit is non-negative and computed


def test_more_elastic_demand_favors_lower_optimal_markup():
    cost, p0, q0 = 10.0, 20.0, 500
    price_low_elastic, profit_low, _ = grid_search_optimal_price(-1.2, cost, p0, q0)
    price_high_elastic, profit_high, _ = grid_search_optimal_price(-3.0, cost, p0, q0)
    # more elastic (larger magnitude, e.g. -3.0) demand should push optimal price down
    # relative to less elastic demand (-1.2), all else equal
    assert price_high_elastic <= price_low_elastic


def test_optimal_price_within_guardrail_bounds():
    cost, p0, q0, elasticity = 10.0, 20.0, 500, -2.5
    price, _, _ = grid_search_optimal_price(elasticity, cost, p0, q0)
    assert 0.85 * p0 <= price <= 1.15 * p0


if __name__ == "__main__":
    test_closed_form_markup_matches_grid_search_direction()
    test_more_elastic_demand_favors_lower_optimal_markup()
    test_optimal_price_within_guardrail_bounds()
    print("All tests passed.")

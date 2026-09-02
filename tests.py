"""
Minimal correctness checks. Run: python tests.py

These guard the two things most likely to silently break a targeting model:
leakage through aggregate features, and an evaluation split that is not
forward in time.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

import evaluate as ev
import features as ft
from generate_data import generate


def test_no_self_leak_in_picker_rate():
    df = generate(20_000, seed=3)
    seen = df[df["picker_prior_picks"] >= 20]
    # A picker's prior rate must never be a perfect read on the current row.
    corr = np.corrcoef(seen["picker_prior_error_rate"], seen["pick_error"])[0, 1]
    assert abs(corr) < 0.25, f"suspiciously high correlation: {corr:.3f}"
    print("ok  picker prior rate does not leak")


def test_time_split_is_ordered():
    df = generate(10_000, seed=4)
    train, test = ft.time_split(df, holdout_frac=0.3)
    assert train["pick_id"].max() < test["pick_id"].min()
    assert 0.25 < len(test) / len(df) < 0.35
    print("ok  split is forward in time")


def test_lift_of_perfect_ranking():
    y = np.array([0] * 96 + [1] * 4)
    perfect = y.astype(float)
    # At a 4% budget a perfect ranking catches everything: lift = 1 / base rate.
    assert abs(ev.lift_at_budget(y, perfect, 0.04) - 25.0) < 1e-6
    assert abs(ev.capture_at_budget(y, perfect, 0.04) - 1.0) < 1e-6
    print("ok  lift and capture behave at the ceiling")


def test_decoys_present():
    df = generate(5_000, seed=5)
    X, _ = ft.feature_matrix(df)
    assert any("tote_color" in c for c in X.columns)
    X2, _ = ft.feature_matrix(df, drop_decoys=True)
    assert not any("tote_color" in c for c in X2.columns)
    print("ok  noise controls present and removable")


if __name__ == "__main__":
    test_no_self_leak_in_picker_rate()
    test_time_split_is_ordered()
    test_lift_of_perfect_ranking()
    test_decoys_present()
    print("\nall checks passed")

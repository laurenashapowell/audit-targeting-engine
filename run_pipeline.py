"""
End-to-end run: generate data, engineer features, train, evaluate, cost out.

    python run_pipeline.py

Writes every artifact in outputs/. Takes about a minute on a laptop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

import cost_model as cm  # noqa: E402
import evaluate as ev  # noqa: E402
import features as ft  # noqa: E402
import model as ml  # noqa: E402
from generate_data import generate  # noqa: E402

OUT = Path("outputs")
DATA = Path("data")
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

NAVY = "#1F3864"
ORANGE = "#D97706"
GRAY = "#8A8A8A"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
    }
)


def main() -> None:
    print(f"backend: {ml.backend_name()}")

    # ---------------------------------------------------------------- data
    df = generate(n_rows=120_000)
    df.to_csv(DATA / "picks_sample.csv.gz", index=False, compression="gzip")
    print(f"generated {len(df):,} picks, error rate {df['pick_error'].mean():.4f}")

    train_df, test_df = ft.time_split(df, holdout_frac=0.30)
    X_train, y_train = ft.feature_matrix(train_df)
    X_test, y_test = ft.feature_matrix(test_df)
    X_test = X_test[X_train.columns]
    print(f"train {len(X_train):,} / test {len(X_test):,} / {X_train.shape[1]} features")

    # ---------------------------------------------------------------- models
    gbm = ml.train_gbm(X_train, y_train)
    logit = ml.train_logistic(X_train, y_train)

    # Ablation: same model class, raw source columns only. The gap between this
    # and the engineered model is what the feature work is actually worth.
    Xr_train, yr_train = ft.raw_matrix(train_df)
    Xr_test, _ = ft.raw_matrix(test_df)
    Xr_test = Xr_test[Xr_train.columns]
    gbm_raw = ml.train_gbm(Xr_train, yr_train)

    scores = {
        "Random sampling": ev.baseline_random(len(y_test)),
        "Incumbent rule (value x qty)": ev.baseline_business_rule(test_df),
        "Boosted, raw features only": ml.predict_proba(gbm_raw, Xr_test),
        "Logistic regression": ml.predict_proba(logit, X_test),
        "Gradient boosted model": ml.predict_proba(gbm, X_test),
    }

    # ---------------------------------------------------------------- metrics
    summaries = [ev.summary(y_test, s, name) for name, s in scores.items()]
    metrics_df = pd.DataFrame(summaries)
    metrics_df.to_csv(OUT / "metrics.csv", index=False)
    print("\n" + metrics_df.round(3).to_string(index=False))

    # ---------------------------------------------------------------- cost
    assumptions = cm.CostAssumptions()
    cost_df = cm.compare(y_test, scores, assumptions)
    cost_df.to_csv(OUT / "cost_analysis.csv", index=False)
    cm.assumptions_table(assumptions).to_csv(OUT / "cost_assumptions.csv", index=False)

    equiv = cm.equivalent_budget(
        y_test,
        scores["Gradient boosted model"],
        scores["Incumbent rule (value x qty)"],
        baseline_budget=0.10,
    )

    headline = {
        "backend": ml.backend_name(),
        "n_test": int(len(y_test)),
        "test_error_rate": float(y_test.mean()),
        "lift_at_5pct_vs_random": float(
            ev.lift_at_budget(y_test, scores["Gradient boosted model"], 0.05)
        ),
        "lift_multiple_vs_incumbent_at_5pct": float(
            ev.lift_at_budget(y_test, scores["Gradient boosted model"], 0.05)
            / max(ev.lift_at_budget(y_test, scores["Incumbent rule (value x qty)"], 0.05), 1e-9)
        ),
        "equivalent_budget": equiv,
    }

    # Annual saving at a fixed 5% inspection budget, model vs incumbent.
    at5 = cost_df[np.isclose(cost_df["budget_frac"], 0.05)].set_index("strategy")
    headline["annual_saving_vs_incumbent_at_5pct"] = float(
        at5.loc["Incumbent rule (value x qty)", "total_cost"]
        - at5.loc["Gradient boosted model", "total_cost"]
    )

    model_costs = cost_df[cost_df["strategy"] == "Gradient boosted model"]
    best_row = model_costs.loc[model_costs["total_cost"].idxmin()]
    headline["cost_optimal_budget_frac"] = float(best_row["budget_frac"])
    headline["cost_at_optimal_budget"] = float(best_row["total_cost"])

    with open(OUT / "headline_results.json", "w") as f:
        json.dump(headline, f, indent=2)
    print("\nheadline:", json.dumps(headline, indent=2))

    # ---------------------------------------------------------------- charts
    _plot_capture(y_test, scores)
    _plot_lift(y_test, scores)
    _plot_cost(cost_df)
    _plot_calibration(y_test, scores["Gradient boosted model"])
    _plot_importance(gbm, X_test, y_test)

    print("\nwrote artifacts to outputs/")


def _plot_capture(y_test, scores):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    styles = {
        "Random sampling": (GRAY, "--"),
        "Incumbent rule (value x qty)": (ORANGE, "-"),
        "Boosted, raw features only": ("#B9C6DC", "-"),
        "Logistic regression": ("#6B8FC4", "-"),
        "Gradient boosted model": (NAVY, "-"),
    }
    for name, s in scores.items():
        c = ev.curve(y_test, s)
        color, ls = styles[name]
        ax.plot(c["budget_frac"] * 100, c["capture_rate"] * 100, label=name,
                color=color, linestyle=ls, linewidth=2.0 if "Gradient" in name else 1.4)
    ax.axvline(5, color="black", linewidth=0.7, alpha=0.4)
    ax.annotate("5% inspection budget", xy=(5, 62), xytext=(14, 48), fontsize=7.5,
                color="black", alpha=0.75,
                arrowprops=dict(arrowstyle="-", color="black", alpha=0.35, lw=0.7))
    ax.set_xlabel("Share of picks inspected (%)")
    ax.set_ylabel("Share of all errors caught (%)")
    ax.set_title("How many errors you catch for a given inspection budget", fontsize=10.5,
                 color=NAVY, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "capture_curve.png")
    plt.close(fig)


def _plot_lift(y_test, scores):
    budgets = [0.01, 0.02, 0.05, 0.10, 0.20]
    names = ["Incumbent rule (value x qty)", "Boosted, raw features only",
             "Logistic regression", "Gradient boosted model"]
    width = 0.20
    x = np.arange(len(budgets))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, name in enumerate(names):
        vals = [ev.lift_at_budget(y_test, scores[name], b) for b in budgets]
        color = [ORANGE, "#B9C6DC", "#6B8FC4", NAVY][i]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=name, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.1f}x",
                    ha="center", fontsize=6.2)
    ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=1)
    ax.text(-0.45, 1.25, "random = 1.0x", fontsize=7, color=GRAY, ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b*100)}%" for b in budgets])
    ax.set_xlabel("Inspection budget")
    ax.set_ylabel("Lift over random sampling")
    ax.set_title("Lift by inspection budget", fontsize=10.5, color=NAVY, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "lift_by_budget.png")
    plt.close(fig)


def _plot_cost(cost_df):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    styles = {
        "Random sampling": (GRAY, "--"),
        "Incumbent rule (value x qty)": (ORANGE, "-"),
        "Gradient boosted model": (NAVY, "-"),
    }
    for name, (color, ls) in styles.items():
        sub = cost_df[cost_df["strategy"] == name].sort_values("budget_frac")
        ax.plot(sub["budget_frac"] * 100, sub["total_cost"] / 1e6, label=name,
                color=color, linestyle=ls, linewidth=2.0 if "Gradient" in name else 1.4)
        if "Gradient" in name:
            best = sub.loc[sub["total_cost"].idxmin()]
            ax.scatter([best["budget_frac"] * 100], [best["total_cost"] / 1e6],
                       color=NAVY, s=55, zorder=5, marker="o")
            ax.annotate(f"optimum ~{best['budget_frac']*100:.0f}%",
                        xy=(best["budget_frac"] * 100, best["total_cost"] / 1e6),
                        xytext=(best["budget_frac"] * 100 + 6, best["total_cost"] / 1e6 + 1.1),
                        fontsize=7.5, color=NAVY,
                        arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.8))
    ax.set_xlabel("Share of picks inspected (%)")
    ax.set_ylabel("Total annual cost, $M\n(inspection labor + missed errors)")
    ax.set_title("Total cost of quality by strategy and budget", fontsize=10.5,
                 color=NAVY, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cost_curve.png")
    plt.close(fig)


def _plot_calibration(y_test, model_scores):
    tbl = ev.calibration_table(y_test, model_scores, bins=10)
    tbl.to_csv(OUT / "calibration.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    lim = max(tbl["mean_predicted"].max(), tbl["observed_rate"].max()) * 1.1
    ax.plot([0, lim], [0, lim], color=GRAY, linestyle="--", linewidth=1, label="perfect")
    ax.scatter(tbl["mean_predicted"], tbl["observed_rate"], color=NAVY, s=42, zorder=3,
               label="score decile")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed error rate")
    ax.set_title("Calibration by score decile", fontsize=10.5, color=NAVY, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png")
    plt.close(fig)


def _plot_importance(gbm, X_test, y_test):
    """
    Permutation importance on the holdout, scored by average precision.

    Permutation is used instead of a native importance attribute because it
    measures effect on the metric we actually care about, and because it works
    identically across both model backends. The decoy columns are the check:
    tote_color, label_printer_id, and day_of_month carry no signal by
    construction, so anything ranking below them is noise.
    """
    from sklearn.inspection import permutation_importance

    sub = min(12_000, len(X_test))
    idx = np.random.default_rng(7).choice(len(X_test), sub, replace=False)
    r = permutation_importance(
        gbm, X_test.iloc[idx], y_test[idx],
        scoring="average_precision", n_repeats=4, random_state=7, n_jobs=-1,
    )
    imp = (
        pd.DataFrame({"feature": X_test.columns, "importance": r.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv(OUT / "feature_importance.csv", index=False)

    top = imp.head(15).iloc[::-1]
    decoys = {"tote_color_gray", "tote_color_red", "tote_color_green",
              "label_printer_id", "day_of_month"}
    colors = [ORANGE if f in decoys else NAVY for f in top["feature"]]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(top["feature"], top["importance"], color=colors)
    ax.set_xlabel("Drop in average precision when shuffled")
    ax.set_title("Permutation importance (orange = known-noise control)",
                 fontsize=10.5, color=NAVY, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png")
    plt.close(fig)

    worst_decoy_rank = max(
        (imp.index[imp["feature"] == d][0] for d in decoys if (imp["feature"] == d).any()),
        default=None,
    )
    print(f"\nlowest-ranked decoy sits at position {worst_decoy_rank} of {len(imp)}")


if __name__ == "__main__":
    main()

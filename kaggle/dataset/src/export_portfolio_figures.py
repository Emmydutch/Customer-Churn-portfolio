"""Export the two portfolio figures referenced during notebook review."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "figures"
NAVY, BLUE, ORANGE = "#17345B", "#377BB5", "#F28E2B"


def export_churn_reasons() -> Path:
    df = pd.read_csv(ROOT / "data" / "processed" / "telco_customer_churn_features.csv")
    churned = df.loc[df.churn_flag.eq(1)]
    category = churned.churn_category.value_counts().rename_axis("category").reset_index(name="customers")
    category["share"] = category.customers / category.customers.sum()
    reasons = churned.churn_reason.value_counts().head(10).sort_values()

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [0.42, 0.58]})
    axes[0].bar(category.category, category.share, color=BLUE)
    axes[0].set_title("Share of Churn by Category", color=NAVY, fontsize=13, pad=12)
    axes[0].set_ylabel("Share of churn", color=NAVY)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].tick_params(axis="x", rotation=18)

    axes[1].barh(reasons.index, reasons.values, color=ORANGE)
    axes[1].set_title("Top 10 Recorded Reasons", color=NAVY, fontsize=13, pad=12)
    axes[1].set_xlabel("Churned customers", color=NAVY)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E8EDF4", linewidth=.8)
        axis.set_axisbelow(True)
    fig.suptitle("Why Customers Churned", color=NAVY, fontsize=17, x=.08, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .96), h_pad=3)
    path = OUTPUT / "why-customers-churned.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def export_cv_tradeoff() -> Path:
    results = pd.read_csv(ROOT / "artifacts" / "modeling" / "cross_validation_results.csv")
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    scatter = ax.scatter(
        results.cv_recall_mean, results.cv_precision_mean,
        s=350 + results.cv_pr_auc_mean * 900,
        c=results.cv_pr_auc_mean, cmap="Blues", edgecolor="white", linewidth=1,
    )
    offsets = {
        "Gradient Boosting": (-95, 42), "Random Forest": (55, 42),
        "Logistic Regression": (65, -38), "Decision Tree": (-90, -38),
        "Dummy Baseline": (0, 35),
    }
    for row in results.itertuples():
        ax.annotate(
            row.model, (row.cv_recall_mean, row.cv_precision_mean),
            xytext=offsets[row.model], textcoords="offset points", ha="center",
            color=NAVY, fontsize=9,
            bbox=dict(boxstyle="round,pad=.25", fc="white", ec="#D8E1EC", alpha=.95),
            arrowprops=dict(arrowstyle="-", color="#8FA3BF", linewidth=.8),
        )
    ax.set_title("Cross-Validated Precision–Recall Tradeoff", loc="left", color=NAVY, fontsize=16, pad=18)
    ax.set_xlabel("Mean recall", color=NAVY)
    ax.set_ylabel("Mean precision", color=NAVY)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlim(-.03, .78)
    ax.set_ylim(-.05, .88)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#E8EDF4", linewidth=.8)
    ax.set_axisbelow(True)
    colorbar = fig.colorbar(scatter, ax=ax, pad=.03)
    colorbar.set_label("PR-AUC", color=NAVY)
    path = OUTPUT / "cross-validated-precision-recall-tradeoff.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for exported in (export_churn_reasons(), export_cv_tradeoff()):
        print(exported.relative_to(ROOT))

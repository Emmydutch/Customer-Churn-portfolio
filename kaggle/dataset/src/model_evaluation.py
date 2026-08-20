"""Threshold selection, calibration, fairness audit, and portfolio scoring."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.modeling import (
    CV_FOLDS,
    MODEL_FEATURES,
    RANDOM_STATE,
    evaluate_probabilities,
)


MINIMUM_RECALL = 0.80
THRESHOLDS = np.round(np.arange(0.05, 0.81, 0.01), 2)
FAIRNESS_ATTRIBUTES = ["gender", "senior_citizen", "age_group"]


def load_evaluation_data(
    feature_path: str | Path,
    split_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the feature data and reproduce the persisted train/test populations."""

    df = pd.read_csv(
        feature_path,
        dtype={"customer_id": "string", "zip_code": "string"},
    )
    split = pd.read_csv(split_path, dtype={"customer_id": "string"})
    if split["customer_id"].duplicated().any() or len(split) != len(df):
        raise ValueError("Persisted split membership does not reconcile with feature data.")
    train_ids = set(split.loc[split["split"].eq("train"), "customer_id"])
    test_ids = set(split.loc[split["split"].eq("test"), "customer_id"])
    train_df = df.loc[df["customer_id"].isin(train_ids)].copy()
    test_df = df.loc[df["customer_id"].isin(test_ids)].copy()
    if len(train_df) + len(test_df) != len(df) or train_ids & test_ids:
        raise AssertionError("Train/test populations overlap or omit customers.")
    return df, train_df, test_df


def generate_oof_probabilities(
    pipeline: object,
    train_df: pd.DataFrame,
) -> np.ndarray:
    """Generate out-of-fold training probabilities for threshold selection."""

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(
        clone(pipeline),
        train_df[MODEL_FEATURES],
        train_df["churn_flag"],
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]


def build_threshold_table(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Evaluate operating metrics across a fixed threshold grid."""

    rows = []
    y_array = np.asarray(y_true)
    for threshold in THRESHOLDS:
        metrics = evaluate_probabilities(y_array, probabilities, float(threshold))
        predictions = probabilities >= threshold
        rows.append(
            {
                **metrics,
                "f2": fbeta_score(y_array, predictions, beta=2, zero_division=0),
                "flagged_customers": int(predictions.sum()),
                "flagged_rate": float(predictions.mean()),
            }
        )
    return pd.DataFrame(rows)


def select_operating_threshold(
    threshold_table: pd.DataFrame,
    minimum_recall: float = MINIMUM_RECALL,
) -> tuple[float, str]:
    """Choose the highest-precision threshold that preserves minimum recall."""

    eligible = threshold_table.loc[threshold_table["recall"].ge(minimum_recall)]
    if eligible.empty:
        selected = threshold_table.sort_values(["recall", "precision"], ascending=False).iloc[0]
        rationale = "No threshold met the recall requirement; selected maximum recall."
    else:
        selected = eligible.sort_values(
            ["precision", "f1", "threshold"], ascending=False
        ).iloc[0]
        rationale = (
            f"Selected the highest-precision out-of-fold threshold with recall at least "
            f"{minimum_recall:.0%}."
        )
    return float(selected["threshold"]), rationale


def build_calibration_report(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Calculate quantile-bin calibration, ECE, slope, intercept, and log loss."""

    y_array = np.asarray(y_true, dtype=int)
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    calibration = pd.DataFrame({"actual": y_array, "probability": clipped})
    calibration["bin"] = pd.qcut(
        calibration["probability"], q=bins, duplicates="drop"
    )
    table = (
        calibration.groupby("bin", observed=True)
        .agg(
            customers=("actual", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_churn_rate=("actual", "mean"),
        )
        .reset_index()
    )
    table["absolute_calibration_error"] = (
        table["mean_predicted_probability"] - table["observed_churn_rate"]
    ).abs()
    expected_calibration_error = float(
        (table["customers"] / len(calibration) * table["absolute_calibration_error"]).sum()
    )
    logit_probability = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibration_model = LogisticRegression(C=1e6, solver="lbfgs").fit(
        logit_probability, y_array
    )
    metrics = {
        "brier_score": float(np.mean((clipped - y_array) ** 2)),
        "log_loss": float(log_loss(y_array, clipped)),
        "expected_calibration_error": expected_calibration_error,
        "calibration_intercept": float(calibration_model.intercept_[0]),
        "calibration_slope": float(calibration_model.coef_[0, 0]),
        "calibration_bins": int(len(table)),
    }
    table["bin"] = table["bin"].astype("string")
    return table, metrics


def build_fairness_audit(
    test_df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit threshold performance across demographic groups excluded from prediction."""

    audit_df = test_df.reset_index(drop=True).copy()
    audit_df["predicted_probability"] = probabilities
    audit_df["predicted_churn"] = probabilities >= threshold
    rows = []
    for attribute in FAIRNESS_ATTRIBUTES:
        for group, subset in audit_df.groupby(attribute, observed=True):
            y_true = subset["churn_flag"].to_numpy()
            predictions = subset["predicted_churn"].astype(int).to_numpy()
            tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
            rows.append(
                {
                    "attribute": attribute,
                    "group": group,
                    "customers": len(subset),
                    "actual_churn_rate": float(y_true.mean()),
                    "selection_rate": float(predictions.mean()),
                    "precision": precision_score(y_true, predictions, zero_division=0),
                    "recall": recall_score(y_true, predictions, zero_division=0),
                    "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
                    "true_negative_rate": tn / (tn + fp) if tn + fp else np.nan,
                    "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
                    "roc_auc": (
                        roc_auc_score(y_true, subset["predicted_probability"])
                        if np.unique(y_true).size == 2
                        else np.nan
                    ),
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp),
                }
            )
    metrics = pd.DataFrame(rows)
    gap_rows = []
    for attribute, subset in metrics.groupby("attribute", observed=True):
        gap_rows.append(
            {
                "attribute": attribute,
                "groups": len(subset),
                "recall_gap": subset["recall"].max() - subset["recall"].min(),
                "false_positive_rate_gap": (
                    subset["false_positive_rate"].max()
                    - subset["false_positive_rate"].min()
                ),
                "selection_rate_gap": (
                    subset["selection_rate"].max() - subset["selection_rate"].min()
                ),
                "minimum_group_size": int(subset["customers"].min()),
            }
        )
    return metrics, pd.DataFrame(gap_rows)


def run_model_evaluation(
    feature_path: str | Path = "data/processed/telco_customer_churn_features.csv",
    model_path: str | Path = "artifacts/modeling/selected_churn_model.joblib",
    split_path: str | Path = "artifacts/modeling/split_membership.csv",
    output_directory: str | Path = "artifacts/evaluation",
) -> dict[str, object]:
    """Select threshold, evaluate holdout, audit fairness, and score portfolio."""

    df, train_df, test_df = load_evaluation_data(feature_path, split_path)
    selected_pipeline = joblib.load(model_path)

    oof_probabilities = generate_oof_probabilities(selected_pipeline, train_df)
    threshold_table = build_threshold_table(train_df["churn_flag"], oof_probabilities)
    threshold, threshold_rationale = select_operating_threshold(threshold_table)

    test_probabilities = selected_pipeline.predict_proba(test_df[MODEL_FEATURES])[:, 1]
    holdout_metrics = evaluate_probabilities(
        test_df["churn_flag"], test_probabilities, threshold
    )
    holdout_metrics["f2"] = fbeta_score(
        test_df["churn_flag"], test_probabilities >= threshold, beta=2
    )
    holdout_metrics["flagged_customers"] = int((test_probabilities >= threshold).sum())
    holdout_metrics["flagged_rate"] = float((test_probabilities >= threshold).mean())

    calibration_table, calibration_metrics = build_calibration_report(
        test_df["churn_flag"], test_probabilities
    )
    fairness_metrics, fairness_gaps = build_fairness_audit(
        test_df, test_probabilities, threshold
    )

    # After evaluation choices are locked, refit the selected pipeline on all
    # labeled records for portfolio scoring and later application use.
    production_pipeline = clone(selected_pipeline)
    production_pipeline.fit(df[MODEL_FEATURES], df["churn_flag"])
    portfolio_probabilities = production_pipeline.predict_proba(df[MODEL_FEATURES])[:, 1]
    portfolio_scores = pd.DataFrame(
        {
            "customer_id": df["customer_id"],
            "customer_status": df["customer_status"],
            "actual_churn": df["churn_flag"],
            "predicted_churn_probability": portfolio_probabilities,
            "high_risk_at_selected_threshold": portfolio_probabilities >= threshold,
            "monthly_charge": df["monthly_charge"],
            "cltv": df["cltv"],
        }
    )
    active_high_risk = portfolio_scores.loc[
        portfolio_scores["customer_status"].ne("Churned")
        & portfolio_scores["high_risk_at_selected_threshold"]
    ]

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "threshold_table": output_dir / "threshold_analysis.csv",
        "holdout_metrics": output_dir / "selected_threshold_holdout_metrics.json",
        "calibration_table": output_dir / "calibration_table.csv",
        "calibration_metrics": output_dir / "calibration_metrics.json",
        "fairness_metrics": output_dir / "fairness_group_metrics.csv",
        "fairness_gaps": output_dir / "fairness_gap_summary.csv",
        "portfolio_scores": output_dir / "portfolio_risk_scores.csv",
        "production_model": output_dir / "production_churn_model.joblib",
        "evaluation_report": output_dir / "model_evaluation_report.json",
    }
    threshold_table.to_csv(paths["threshold_table"], index=False)
    paths["holdout_metrics"].write_text(
        json.dumps({key: float(value) for key, value in holdout_metrics.items()}, indent=2),
        encoding="utf-8",
    )
    calibration_table.to_csv(paths["calibration_table"], index=False)
    paths["calibration_metrics"].write_text(
        json.dumps(calibration_metrics, indent=2), encoding="utf-8"
    )
    fairness_metrics.to_csv(paths["fairness_metrics"], index=False)
    fairness_gaps.to_csv(paths["fairness_gaps"], index=False)
    portfolio_scores.to_csv(paths["portfolio_scores"], index=False)
    joblib.dump(production_pipeline, paths["production_model"])

    report = {
        "selected_threshold": threshold,
        "threshold_selection_population": "out-of-fold training predictions",
        "minimum_recall_requirement": MINIMUM_RECALL,
        "threshold_rationale": threshold_rationale,
        "holdout_metrics": {key: float(value) for key, value in holdout_metrics.items()},
        "calibration_metrics": calibration_metrics,
        "active_customers": int(df["active_flag"].sum()),
        "active_high_risk_customers": int(len(active_high_risk)),
        "active_high_risk_rate": float(len(active_high_risk) / df["active_flag"].sum()),
        "active_high_risk_monthly_charge_exposure": float(active_high_risk["monthly_charge"].sum()),
        "active_high_risk_cltv": float(active_high_risk["cltv"].sum()),
        "fairness_note": (
            "Demographics are excluded from prediction and used only for subgroup auditing. "
            "Observed gaps require monitoring and do not by themselves establish discrimination."
        ),
    }
    paths["evaluation_report"].write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    return {
        "selected_threshold": threshold,
        "threshold_rationale": threshold_rationale,
        "threshold_table": threshold_table,
        "holdout_metrics": holdout_metrics,
        "test_df": test_df,
        "test_probabilities": test_probabilities,
        "calibration_table": calibration_table,
        "calibration_metrics": calibration_metrics,
        "fairness_metrics": fairness_metrics,
        "fairness_gaps": fairness_gaps,
        "portfolio_scores": portfolio_scores,
        "report": report,
        "paths": {name: path.resolve() for name, path in paths.items()},
    }


if __name__ == "__main__":
    result = run_model_evaluation()
    print(f"Selected threshold: {result['selected_threshold']:.2f}")
    print(result["threshold_rationale"])
    print(json.dumps(result["report"], indent=2))
    for name, path in result["paths"].items():
        print(f"{name}: {path}")

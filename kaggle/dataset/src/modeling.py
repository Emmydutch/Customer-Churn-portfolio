"""Reproducible, leakage-safe churn model development pipeline."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
PR_AUC_SIMPLICITY_TOLERANCE = 0.02

# Operational predictors are plausibly available before churn. Sensitive
# demographics, geography, outcomes, external scores, unverified CLTV and
# satisfaction, and duplicated accumulated revenue fields are excluded.
NUMERIC_FEATURES = [
    "number_of_dependents",
    "number_of_referrals",
    "tenure_in_months",
    "avg_monthly_long_distance_charges",
    "avg_monthly_gb_download",
    "monthly_charge",
    "total_refunds",
    "total_extra_data_charges",
]

CATEGORICAL_FEATURES = [
    "married",
    "offer",
    "phone_service",
    "multiple_lines",
    "internet_type",
    "online_security",
    "online_backup",
    "device_protection_plan",
    "premium_tech_support",
    "streaming_tv",
    "streaming_movies",
    "streaming_music",
    "unlimited_data",
    "contract",
    "paperless_billing",
    "payment_method",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EXPLICITLY_EXCLUDED = {
    "customer_id",
    "gender",
    "age",
    "under_30",
    "senior_citizen",
    "dependents",
    "country",
    "state",
    "city",
    "zip_code",
    "latitude",
    "longitude",
    "population",
    "quarter",
    "referred_a_friend",
    "internet_service",
    "total_charges",
    "total_long_distance_charges",
    "total_revenue",
    "satisfaction_score",
    "customer_status",
    "churn_label",
    "churn_flag",
    "active_flag",
    "churn_score",
    "cltv",
    "churn_category",
    "churn_reason",
    "tenure_group",
    "age_group",
    "monthly_charge_band",
    "customer_value_segment",
    "service_count",
    "protection_support_service_count",
    "referral_group",
    "contract_risk_group",
    "descriptive_risk_points",
    "descriptive_risk_segment",
    "avg_revenue_per_tenure_month",
    "customer_engagement_profile",
}


def build_preprocessor() -> ColumnTransformer:
    """Create train-fitted preprocessing for numerical and categorical fields."""

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_candidate_models() -> dict[str, object]:
    """Create deterministic candidate estimators in increasing complexity."""

    return {
        "Dummy Baseline": DummyClassifier(strategy="prior", random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(
            max_iter=2_000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=25,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        ),
    }


def build_pipeline(estimator: object) -> Pipeline:
    """Combine preprocessing and estimator so transformations stay inside CV."""

    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def split_model_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create one reproducible stratified train/test split and retain customer IDs."""

    missing = sorted(set(MODEL_FEATURES + ["churn_flag", "customer_id"]) - set(df.columns))
    if missing:
        raise ValueError(f"Model data is missing required columns: {missing}")
    X = df[MODEL_FEATURES].copy()
    y = df["churn_flag"].astype("int8")
    ids = df["customer_id"].astype("string")
    return train_test_split(
        X,
        y,
        ids,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def evaluate_probabilities(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, float | int]:
    """Calculate probability, classification, and confusion-matrix metrics."""

    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def cross_validate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """Evaluate all candidates on identical stratified training folds."""

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "neg_brier": "neg_brier_score",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }
    rows: list[dict[str, object]] = []
    for model_name, estimator in build_candidate_models().items():
        results = cross_validate(
            build_pipeline(estimator),
            X_train,
            y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )
        row: dict[str, object] = {"model": model_name}
        for metric in scoring:
            values = results[f"test_{metric}"]
            if metric == "neg_brier":
                values = -values
                output_metric = "brier_score"
            else:
                output_metric = metric
            row[f"cv_{output_metric}_mean"] = float(np.mean(values))
            row[f"cv_{output_metric}_std"] = float(np.std(values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cv_pr_auc_mean", ascending=False).reset_index(drop=True)


def choose_model(cv_results: pd.DataFrame) -> tuple[str, str]:
    """Prefer the simplest model within a predefined PR-AUC tolerance of best."""

    non_dummy = cv_results.loc[cv_results["model"].ne("Dummy Baseline")].copy()
    best = non_dummy.sort_values("cv_pr_auc_mean", ascending=False).iloc[0]
    cutoff = best["cv_pr_auc_mean"] - PR_AUC_SIMPLICITY_TOLERANCE
    simplicity_order = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
    ]
    eligible = set(non_dummy.loc[non_dummy["cv_pr_auc_mean"].ge(cutoff), "model"])
    selected = next(name for name in simplicity_order if name in eligible)
    rationale = (
        f"Selected {selected} using a predefined {PR_AUC_SIMPLICITY_TOLERANCE:.2f} "
        f"practical-equivalence tolerance on cross-validated PR-AUC. "
        f"Best mean PR-AUC was {best['cv_pr_auc_mean']:.4f} ({best['model']}); "
        f"eligible cutoff was {cutoff:.4f}."
    )
    return selected, rationale


def fit_and_test_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, np.ndarray]]:
    """Fit candidates on all training data and evaluate the untouched test set."""

    rows = []
    fitted: dict[str, Pipeline] = {}
    probabilities: dict[str, np.ndarray] = {}
    for model_name, estimator in build_candidate_models().items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        probability = pipeline.predict_proba(X_test)[:, 1]
        rows.append({"model": model_name, **evaluate_probabilities(y_test, probability)})
        fitted[model_name] = pipeline
        probabilities[model_name] = probability
    results = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    return results, fitted, probabilities


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Extract signed coefficients or impurity importances from a fitted pipeline."""

    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    names = [
        name.replace("numeric__", "").replace("categorical__", "") for name in names
    ]
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "coef_"):
        signed = estimator.coef_[0]
        importance = np.abs(signed)
        measure = "standardized_log_odds_coefficient"
    elif hasattr(estimator, "feature_importances_"):
        signed = estimator.feature_importances_
        importance = estimator.feature_importances_
        measure = "impurity_importance"
    else:
        return pd.DataFrame(columns=["feature", "signed_effect", "importance", "measure"])
    return (
        pd.DataFrame(
            {
                "feature": names,
                "signed_effect": signed,
                "importance": importance,
                "measure": measure,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_model_suite(
    feature_path: str | Path = "data/processed/telco_customer_churn_features.csv",
    artifact_directory: str | Path = "artifacts/modeling",
) -> dict[str, object]:
    """Run cross-validation, holdout evaluation, model selection, and export."""

    df = pd.read_csv(
        feature_path,
        dtype={"customer_id": "string", "zip_code": "string"},
    )
    uncovered = set(df.columns) - set(MODEL_FEATURES) - EXPLICITLY_EXCLUDED
    if uncovered:
        raise AssertionError(f"Model feature governance is incomplete: {sorted(uncovered)}")

    X_train, X_test, y_train, y_test, id_train, id_test = split_model_data(df)
    cv_results = cross_validate_candidates(X_train, y_train)
    selected_name, selection_rationale = choose_model(cv_results)
    test_results, fitted_models, test_probabilities = fit_and_test_candidates(
        X_train, y_train, X_test, y_test
    )
    selected_pipeline = fitted_models[selected_name]
    selected_probabilities = test_probabilities[selected_name]
    importance = extract_feature_importance(selected_pipeline)

    artifact_dir = Path(artifact_directory)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": artifact_dir / "selected_churn_model.joblib",
        "cv_results": artifact_dir / "cross_validation_results.csv",
        "test_results": artifact_dir / "holdout_test_results.csv",
        "test_predictions": artifact_dir / "holdout_predictions.csv",
        "feature_importance": artifact_dir / "selected_model_feature_importance.csv",
        "split_membership": artifact_dir / "split_membership.csv",
        "metadata": artifact_dir / "model_metadata.json",
    }

    joblib.dump(selected_pipeline, paths["model"])
    cv_results.to_csv(paths["cv_results"], index=False)
    test_results.to_csv(paths["test_results"], index=False)
    importance.to_csv(paths["feature_importance"], index=False)
    pd.DataFrame(
        {
            "customer_id": id_test.reset_index(drop=True),
            "actual_churn": y_test.reset_index(drop=True),
            "predicted_probability": selected_probabilities,
            "predicted_class_at_0_50": (selected_probabilities >= 0.50).astype(int),
        }
    ).to_csv(paths["test_predictions"], index=False)
    pd.concat(
        [
            pd.DataFrame({"customer_id": id_train.reset_index(drop=True), "split": "train"}),
            pd.DataFrame({"customer_id": id_test.reset_index(drop=True), "split": "test"}),
        ],
        ignore_index=True,
    ).to_csv(paths["split_membership"], index=False)

    selected_test = test_results.set_index("model").loc[selected_name].to_dict()
    metadata = {
        "selected_model": selected_name,
        "selection_rationale": selection_rationale,
        "selection_metric": "cross-validated average precision (PR-AUC)",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "pr_auc_simplicity_tolerance": PR_AUC_SIMPLICITY_TOLERANCE,
        "training_customers": int(len(X_train)),
        "test_customers": int(len(X_test)),
        "training_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "explicitly_excluded_features": sorted(EXPLICITLY_EXCLUDED),
        "default_threshold": 0.50,
        "selected_test_metrics": {key: float(value) for key, value in selected_test.items()},
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "selected_model_name": selected_name,
        "selection_rationale": selection_rationale,
        "selected_pipeline": selected_pipeline,
        "cv_results": cv_results,
        "test_results": test_results,
        "feature_importance": importance,
        "paths": {name: path.resolve() for name, path in paths.items()},
        "train_size": len(X_train),
        "test_size": len(X_test),
        "y_train": y_train,
        "y_test": y_test,
        "test_probabilities": selected_probabilities,
    }


if __name__ == "__main__":
    result = train_model_suite()
    print(result["selection_rationale"])
    print(result["test_results"].to_string(index=False))
    for name, path in result["paths"].items():
        print(f"{name}: {path}")

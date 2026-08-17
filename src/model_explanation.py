"""Global and customer-level explanations for the selected churn model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.modeling import MODEL_FEATURES, RANDOM_STATE


NUMERIC_PROFILE_FEATURES = ["tenure_in_months", "monthly_charge", "number_of_referrals"]
CATEGORICAL_PROFILE_FEATURES = [
    "contract",
    "internet_type",
    "premium_tech_support",
    "online_security",
    "payment_method",
]


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1 / (1 + np.exp(-np.asarray(value)))


def _clean_transformed_name(name: str) -> str:
    return name.replace("numeric__", "").replace("categorical__", "")


def _original_feature(term: str) -> str:
    """Map a transformed one-hot term back to its original model feature."""

    for feature in sorted(MODEL_FEATURES, key=len, reverse=True):
        if term == feature or term.startswith(f"{feature}_"):
            return feature
    raise ValueError(f"Cannot map transformed term to original feature: {term}")


def build_coefficient_table(production_pipeline: object) -> pd.DataFrame:
    """Return signed logistic coefficients and conditional odds multipliers."""

    preprocessor = production_pipeline.named_steps["preprocess"]
    model = production_pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise TypeError("Coefficient explanations require a fitted linear classifier.")
    terms = [_clean_transformed_name(name) for name in preprocessor.get_feature_names_out()]
    coefficients = model.coef_[0]
    result = pd.DataFrame(
        {
            "term": terms,
            "original_feature": [_original_feature(term) for term in terms],
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
            "conditional_odds_multiplier": np.exp(coefficients),
            "direction": np.where(
                coefficients > 0, "Higher modeled churn odds", "Lower modeled churn odds"
            ),
        }
    )
    result["interpretation_unit"] = np.where(
        result["term"].isin(
            ["number_of_dependents", "number_of_referrals", "tenure_in_months",
             "avg_monthly_long_distance_charges", "avg_monthly_gb_download",
             "monthly_charge", "total_refunds", "total_extra_data_charges"]
        ),
        "One training-standard-deviation increase",
        "Category relative to the encoder reference",
    )
    return result.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)


def build_permutation_importance(
    selected_pipeline: object,
    test_df: pd.DataFrame,
    repeats: int = 20,
) -> pd.DataFrame:
    """Measure holdout PR-AUC loss after shuffling each original input feature."""

    result = permutation_importance(
        selected_pipeline,
        test_df[MODEL_FEATURES],
        test_df["churn_flag"],
        scoring="average_precision",
        n_repeats=repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "pr_auc_importance_mean": result.importances_mean,
            "pr_auc_importance_std": result.importances_std,
            "positive_repeats": (result.importances > 0).sum(axis=1),
            "repeats": repeats,
        }
    )
    importance["importance_ci_lower_approx"] = (
        importance["pr_auc_importance_mean"] - 1.96 * importance["pr_auc_importance_std"]
    )
    importance["importance_ci_upper_approx"] = (
        importance["pr_auc_importance_mean"] + 1.96 * importance["pr_auc_importance_std"]
    )
    return importance.sort_values("pr_auc_importance_mean", ascending=False).reset_index(drop=True)


def build_marginal_profiles(
    production_pipeline: object,
    background_df: pd.DataFrame,
) -> pd.DataFrame:
    """Average model predictions while varying one feature at a time."""

    rows: list[dict[str, object]] = []
    background = background_df[MODEL_FEATURES].copy()
    baseline_prediction = float(
        production_pipeline.predict_proba(background)[:, 1].mean()
    )
    for feature in NUMERIC_PROFILE_FEATURES:
        lower, upper = background[feature].quantile([0.05, 0.95])
        values = np.unique(np.round(np.linspace(lower, upper, 15), 2))
        for value in values:
            scenario = background.copy()
            scenario[feature] = value
            mean_probability = float(
                production_pipeline.predict_proba(scenario)[:, 1].mean()
            )
            rows.append(
                {
                    "feature": feature,
                    "feature_type": "numeric",
                    "value": str(value),
                    "numeric_value": float(value),
                    "mean_predicted_probability": mean_probability,
                    "change_from_background_mean": mean_probability - baseline_prediction,
                }
            )
    for feature in CATEGORICAL_PROFILE_FEATURES:
        for value in sorted(background[feature].dropna().astype(str).unique()):
            scenario = background.copy()
            scenario[feature] = value
            mean_probability = float(
                production_pipeline.predict_proba(scenario)[:, 1].mean()
            )
            rows.append(
                {
                    "feature": feature,
                    "feature_type": "categorical",
                    "value": value,
                    "numeric_value": np.nan,
                    "mean_predicted_probability": mean_probability,
                    "change_from_background_mean": mean_probability - baseline_prediction,
                }
            )
    return pd.DataFrame(rows)


def build_local_contributions(
    production_pipeline: object,
    background_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """Calculate exact mean-centered additive log-odds contributions."""

    X = background_df[MODEL_FEATURES]
    preprocessor = production_pipeline.named_steps["preprocess"]
    model = production_pipeline.named_steps["model"]
    transformed = np.asarray(preprocessor.transform(X), dtype=float)
    coefficients = model.coef_[0]
    terms = [_clean_transformed_name(name) for name in preprocessor.get_feature_names_out()]
    groups = [_original_feature(term) for term in terms]
    transformed_mean = transformed.mean(axis=0)
    base_logit = float(model.intercept_[0] + transformed_mean @ coefficients)
    base_probability = float(_sigmoid(base_logit))
    centered_contributions = (transformed - transformed_mean) * coefficients

    grouped = pd.DataFrame(index=background_df.index)
    for feature in MODEL_FEATURES:
        indices = [index for index, group in enumerate(groups) if group == feature]
        grouped[feature] = centered_contributions[:, indices].sum(axis=1)

    reconstructed_logit = base_logit + grouped.sum(axis=1).to_numpy()
    reconstructed_probability = _sigmoid(reconstructed_logit)
    model_probability = production_pipeline.predict_proba(X)[:, 1]
    if not np.allclose(reconstructed_probability, model_probability, atol=1e-10):
        raise AssertionError("Local contributions do not reconstruct model probabilities.")

    background_metadata = pd.DataFrame(
        {
            "term": terms,
            "original_feature": groups,
            "background_transformed_mean": transformed_mean,
            "coefficient": coefficients,
        }
    )
    return grouped, base_probability, background_metadata


def explain_customer(
    customer_id: str,
    customer_df: pd.DataFrame,
    grouped_contributions: pd.DataFrame,
    probabilities: np.ndarray,
    base_probability: float,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return the strongest positive and protective factors for one customer."""

    matches = customer_df.index[customer_df["customer_id"].eq(customer_id)]
    if len(matches) != 1:
        raise ValueError(f"Expected one customer for {customer_id}; found {len(matches)}")
    index = matches[0]
    contributions = grouped_contributions.loc[index]
    positive = contributions.loc[contributions.gt(0)].nlargest(top_n)
    negative = contributions.loc[contributions.lt(0)].nsmallest(top_n)
    selected = pd.concat([positive, negative])
    explanation = selected.rename("log_odds_contribution").to_frame().reset_index()
    explanation = explanation.rename(columns={"index": "feature"})
    explanation["effect"] = np.where(
        explanation["log_odds_contribution"].gt(0),
        "Raises risk relative to portfolio background",
        "Lowers risk relative to portfolio background",
    )
    explanation["odds_multiplier_for_contribution"] = np.exp(
        explanation["log_odds_contribution"]
    )
    explanation["customer_id"] = customer_id
    explanation["predicted_probability"] = float(probabilities[index])
    explanation["background_probability"] = base_probability
    explanation["absolute_contribution"] = explanation["log_odds_contribution"].abs()
    return explanation.sort_values("absolute_contribution", ascending=False).reset_index(drop=True)


def build_active_customer_explanations(
    customer_df: pd.DataFrame,
    grouped_contributions: pd.DataFrame,
    probabilities: np.ndarray,
    base_probability: float,
    top_n_each_direction: int = 3,
) -> pd.DataFrame:
    """Export compact explanations for every currently active customer."""

    active = customer_df.loc[customer_df["active_flag"].eq(1)]
    parts = []
    for index, row in active.iterrows():
        parts.append(
            explain_customer(
                row["customer_id"],
                customer_df,
                grouped_contributions,
                probabilities,
                base_probability,
                top_n=top_n_each_direction,
            )
        )
    return pd.concat(parts, ignore_index=True)


def run_model_explanation(
    feature_path: str | Path = "data/processed/telco_customer_churn_features.csv",
    selected_model_path: str | Path = "artifacts/modeling/selected_churn_model.joblib",
    production_model_path: str | Path = "artifacts/evaluation/production_churn_model.joblib",
    split_path: str | Path = "artifacts/modeling/split_membership.csv",
    output_directory: str | Path = "artifacts/explanations",
) -> dict[str, object]:
    """Build and export global, profile, and customer-level explanations."""

    df = pd.read_csv(
        feature_path,
        dtype={"customer_id": "string", "zip_code": "string"},
    ).reset_index(drop=True)
    split = pd.read_csv(split_path, dtype={"customer_id": "string"})
    test_ids = set(split.loc[split["split"].eq("test"), "customer_id"])
    test_df = df.loc[df["customer_id"].isin(test_ids)].copy()
    selected_pipeline = joblib.load(selected_model_path)
    production_pipeline = joblib.load(production_model_path)

    coefficient_table = build_coefficient_table(production_pipeline)
    permutation_table = build_permutation_importance(selected_pipeline, test_df)
    marginal_profiles = build_marginal_profiles(production_pipeline, test_df)
    grouped_contributions, base_probability, background_metadata = (
        build_local_contributions(production_pipeline, df)
    )
    probabilities = production_pipeline.predict_proba(df[MODEL_FEATURES])[:, 1]
    active_explanations = build_active_customer_explanations(
        df,
        grouped_contributions,
        probabilities,
        base_probability,
    )
    active_indices = df.index[df["active_flag"].eq(1)]
    example_index = active_indices[np.argmax(probabilities[active_indices])]
    example_customer_id = str(df.loc[example_index, "customer_id"])
    example_explanation = explain_customer(
        example_customer_id,
        df,
        grouped_contributions,
        probabilities,
        base_probability,
        top_n=6,
    )

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "coefficients": output_dir / "logistic_coefficients.csv",
        "permutation_importance": output_dir / "global_permutation_importance.csv",
        "marginal_profiles": output_dir / "marginal_prediction_profiles.csv",
        "active_explanations": output_dir / "active_customer_explanations.csv",
        "example_explanation": output_dir / "example_customer_explanation.csv",
        "background_metadata": output_dir / "explanation_background.csv",
        "metadata": output_dir / "explanation_metadata.json",
    }
    coefficient_table.to_csv(paths["coefficients"], index=False)
    permutation_table.to_csv(paths["permutation_importance"], index=False)
    marginal_profiles.to_csv(paths["marginal_profiles"], index=False)
    active_explanations.to_csv(paths["active_explanations"], index=False)
    example_explanation.to_csv(paths["example_explanation"], index=False)
    background_metadata.to_csv(paths["background_metadata"], index=False)
    metadata = {
        "explanation_model": "Production-refitted logistic regression pipeline",
        "global_importance_method": "Holdout permutation importance using PR-AUC loss",
        "local_method": "Exact mean-centered additive logistic log-odds decomposition",
        "background_population": "All 7,043 customer records",
        "background_probability": base_probability,
        "example_customer_id": example_customer_id,
        "example_probability": float(probabilities[example_index]),
        "active_customers_explained": int(df["active_flag"].sum()),
        "shap_status": (
            "SHAP package not installed; exact linear contribution decomposition used. "
            "No approximation is required for logistic regression."
        ),
        "interpretation_warning": (
            "Explanations describe model behavior and conditional associations, not causal effects."
        ),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "coefficient_table": coefficient_table,
        "permutation_importance": permutation_table,
        "marginal_profiles": marginal_profiles,
        "grouped_contributions": grouped_contributions,
        "base_probability": base_probability,
        "active_explanations": active_explanations,
        "example_customer_id": example_customer_id,
        "example_explanation": example_explanation,
        "metadata": metadata,
        "paths": {name: path.resolve() for name, path in paths.items()},
    }


if __name__ == "__main__":
    result = run_model_explanation()
    print(json.dumps(result["metadata"], indent=2))
    print("\nTop global permutation importance:")
    print(result["permutation_importance"].head(12).to_string(index=False))
    print("\nExample customer explanation:")
    print(result["example_explanation"].to_string(index=False))
    for name, path in result["paths"].items():
        print(f"{name}: {path}")

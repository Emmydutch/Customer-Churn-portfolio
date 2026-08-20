"""Build the Kaggle dataset and notebook packages from canonical project files."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KAGGLE_ROOT = ROOT / "kaggle"
DATASET_DIR = KAGGLE_ROOT / "dataset"
NOTEBOOK_PATH = KAGGLE_ROOT / "telecom-churn-prescriptive-retention.ipynb"
SOURCE_NOTEBOOK = ROOT / "customer churn.ipynb"


def markdown_cell(text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def build_dataset_package() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    source_target = DATASET_DIR / "src"
    if source_target.exists():
        shutil.rmtree(source_target)
    shutil.copytree(
        ROOT / "src",
        source_target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    for filename in ["TelcoCustomerChurn.csv", "DATASET_NOTICE.md", "LICENSE"]:
        shutil.copy2(ROOT / filename, DATASET_DIR / filename)

    metadata = {
        "title": "Telecom Churn Portfolio Data and Source Modules",
        "subtitle": "Fictional telecom customer data with reproducible portfolio analysis modules",
        "description": (
            "Companion input package for Emmanuel Onuoha's Telecom Customer Churn notebook. "
            "The fictional IBM/Maven source dataset is identified as Public Domain. Original "
            "Python modules are copyright Emmanuel Onuoha and released under the included MIT License."
        ),
        "id": "onuohaemmanuel/telecom-customer-churn-portfolio-data",
        "licenses": [{"name": "other"}],
        "resources": [
            {
                "path": "TelcoCustomerChurn.csv",
                "description": "Fictional public-domain telecom customer churn source data.",
            }
        ],
    }
    (DATASET_DIR / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (DATASET_DIR / "README.md").write_text(
        """# Telecom Customer Churn Portfolio Input Package

This package supports the Kaggle notebook **Telecom Customer Churn: From Prediction to Prescriptive Retention Decisions** by Emmanuel Onuoha.

## Contents

- `TelcoCustomerChurn.csv`: fictional IBM Cognos Analytics telecom dataset distributed by Maven Analytics and identified as Public Domain.
- `src/`: original reproducible Python analysis modules developed by Emmanuel Onuoha under the MIT License.
- `DATASET_NOTICE.md`: source, attribution, redistribution, and responsible-use details.
- `LICENSE`: licence for the original project code.

The dataset describes fictional customers. It must not be interpreted as real subscriber information.
""",
        encoding="utf-8",
    )


def build_notebook() -> None:
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    notebook = copy.deepcopy(notebook)

    for cell in notebook["cells"]:
        cell["source"] = [line.replace("\ufffd", "—") for line in cell.get("source", [])]
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    notebook["cells"][0] = markdown_cell(
        """# Telecom Customer Churn: From Prediction to Prescriptive Retention Decisions

### An end-to-end business analytics and machine-learning case study

**Developed and Designed by Emmanuel Onuoha**  
Kaggle: [onuohaemmanuel](https://www.kaggle.com/onuohaemmanuel)  
GitHub: [Customer-Churn-portfolio](https://github.com/Emmydutch/Customer-Churn-portfolio)  
Live dashboard: [Customer Churn Intelligence](https://customer-churn-portfolio-z3hvzwkxs3yu8zvenx9yuk.streamlit.app/)

This notebook moves from business definition and data quality through leakage-safe modeling, explainability, retention economics, and a transparent budget-constrained decision policy. All financial outputs are planning scenarios rather than realized causal impact.
"""
    )
    introduction = markdown_cell(
        """## How to Read This Notebook

The analysis is organized as a decision journey:

1. Define the business problem and success criteria.
2. Audit, clean, and document the source data.
3. Engineer business-readable customer features.
4. Identify churn patterns, cohorts, and statistically supported findings.
5. Prevent target leakage and build an interpretable prediction pipeline.
6. Select an operating threshold and evaluate calibration and subgroup performance.
7. Explain portfolio-level and individual predictions.
8. Translate risk into retention scenarios and prescriptive campaign decisions.

The attached Kaggle dataset contains the fictional source CSV and the original reusable Python modules. The next cell locates those inputs and creates a writable working directory so the notebook can run from top to bottom on Kaggle or from the GitHub project locally.
"""
    )
    bootstrap = code_cell(
        '''from pathlib import Path
import os
import shutil
import sys

KAGGLE_INPUT = Path("/kaggle/input")
if KAGGLE_INPUT.exists():
    matches = list(KAGGLE_INPUT.rglob("TelcoCustomerChurn.csv"))
    if not matches:
        raise FileNotFoundError(
            "Attach the 'Telecom Churn Portfolio Data and Source Modules' dataset before running."
        )
    package_root = matches[0].parent
    project_root = Path("/kaggle/working/customer-churn-project")
    project_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(package_root / "TelcoCustomerChurn.csv", project_root / "TelcoCustomerChurn.csv")
    shutil.copytree(package_root / "src", project_root / "src", dirs_exist_ok=True)
else:
    project_root = Path.cwd().resolve()
    if not (project_root / "TelcoCustomerChurn.csv").exists():
        project_root = project_root.parent

os.chdir(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Working directory: {project_root}")
print(f"Source data available: {(project_root / 'TelcoCustomerChurn.csv').exists()}")
print(f"Reusable modules available: {(project_root / 'src').exists()}")
'''
    )
    notebook["cells"][1:1] = [introduction, bootstrap]

    decision_markdown = markdown_cell(
        """# Stage 14 — Prescriptive Retention Decision Centre

Prediction identifies customers with elevated pre-churn risk. The prescriptive layer asks a different question:

> Given limited budget and contact capacity, which economically positive interventions should be prioritized?

For each eligible active customer, the policy estimates probability-weighted incremental saves, retained gross margin, expected contact and offer costs, net benefit, and ROI. It excludes non-positive expected-value cases, ranks the remainder by the selected objective, and admits customers while budget and capacity remain.

The policy is deterministic and auditable, but it is **not a causal uplift model**. Its incremental save rate is an explicit assumption that must be replaced with treatment-versus-control evidence from a campaign pilot.
"""
    )
    decision_code = code_cell(
        '''from src.decision_engine import (
    score_retention_decisions,
    select_decision_portfolio,
    summarize_decision_portfolio,
)

decision_assumptions = {
    "reach_rate": 0.75,
    "acceptance_rate": 0.50,
    "incremental_save_rate": 0.35,
    "retention_horizon_months": 12,
    "gross_margin_rate": 0.70,
    "contact_cost_per_target": 4.0,
    "offer_cost_multiplier": 1.0,
}
decision_scores = score_retention_decisions(campaign_population, **decision_assumptions)
selected_decisions, ranked_decisions = select_decision_portfolio(
    decision_scores,
    budget=15_000,
    capacity=500,
    objective="Maximize expected net benefit",
)
decision_summary = summarize_decision_portfolio(selected_decisions, ranked_decisions)
display(pd.Series(decision_summary, name="Default decision policy"))

decision_actions = (
    selected_decisions.groupby("primary_intervention", as_index=False)
    .agg(
        customers=("customer_id", "size"),
        expected_cost=("expected_campaign_cost", "sum"),
        expected_saves=("expected_incremental_saves", "sum"),
        expected_net_benefit=("expected_net_benefit", "sum"),
    )
    .sort_values("expected_net_benefit", ascending=False)
)
display(decision_actions.round(2))

fig = px.bar(
    decision_actions.sort_values("expected_net_benefit"),
    x="expected_net_benefit",
    y="primary_intervention",
    orientation="h",
    color="expected_saves",
    color_continuous_scale="Blues",
    title="Budget-Constrained Expected Net Benefit by Intervention",
)
fig.update_xaxes(title="Expected net benefit ($)", tickprefix="$")
fig.update_yaxes(title=None)
fig.update_layout(template="plotly_white", height=520)
fig.show()
'''
    )
    decision_checks = code_cell(
        '''stage_14_checks = pd.Series({
    "No more than 500 customers selected": len(selected_decisions) <= 500,
    "Expected campaign cost stays within $15,000": selected_decisions["expected_campaign_cost"].sum() <= 15_000,
    "Every selected customer has positive expected net benefit": selected_decisions["expected_net_benefit"].gt(0).all(),
    "Customer identifiers are not required in the public decision display": "customer_id" not in decision_actions.columns,
    "Decision ranks are consecutive": selected_decisions["decision_rank"].tolist() == list(range(1, len(selected_decisions) + 1)),
})
display(stage_14_checks.to_frame("Passed"))
assert stage_14_checks.all(), "At least one Stage 14 decision-policy validation failed."
'''
    )
    conclusion = markdown_cell(
        """## Stage 14 Deliverable — Prescriptive Retention Decision Policy

Under the documented default assumptions, the policy selects up to 500 positive expected-value customers within a $15,000 expected budget. It connects churn probability to next-best actions without exposing customer identifiers in the public decision summary.

### Responsible decision boundary

- Do not treat predicted risk as proof that a customer will churn.
- Do not claim expected saves or net benefit as realized results.
- Do not use age, gender, or another protected characteristic as a targeting objective.
- Validate incremental lift through randomized treatment and control groups.
- Monitor selection, acceptance, retention, complaints, and outcomes across customer groups.
- Keep human review and customer welfare ahead of automated ranking.

For the interactive eight-page implementation, open the [live Streamlit dashboard](https://customer-churn-portfolio-z3hvzwkxs3yu8zvenx9yuk.streamlit.app/).

---

**Developed and Designed by Emmanuel Onuoha**
"""
    )
    notebook["cells"].extend([decision_markdown, decision_code, decision_checks, conclusion])
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"emmanuel-{index:03d}"
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def write_kernel_metadata() -> None:
    metadata = {
        "id": "onuohaemmanuel/telecom-churn-prescriptive-retention",
        "title": "Telecom Churn: Prescriptive Retention Decisions",
        "code_file": NOTEBOOK_PATH.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": ["onuohaemmanuel/telecom-customer-churn-portfolio-data"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (KAGGLE_ROOT / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    KAGGLE_ROOT.mkdir(parents=True, exist_ok=True)
    build_dataset_package()
    build_notebook()
    write_kernel_metadata()
    print(f"Kaggle notebook: {NOTEBOOK_PATH}")
    print(f"Kaggle dataset package: {DATASET_DIR}")
    print(f"Kernel metadata: {KAGGLE_ROOT / 'kernel-metadata.json'}")

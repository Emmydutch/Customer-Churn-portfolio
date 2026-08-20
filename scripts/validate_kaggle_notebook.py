"""Execute the generated Kaggle notebook in an isolated local workspace."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
KAGGLE_ROOT = ROOT / "kaggle"
NOTEBOOK_PATH = KAGGLE_ROOT / "telecom-churn-prescriptive-retention.ipynb"
DATASET_DIR = KAGGLE_ROOT / "dataset"
WORKSPACE = KAGGLE_ROOT / ".validation_workspace"


def remove_validation_workspace() -> None:
    if not WORKSPACE.exists():
        return
    resolved = WORKSPACE.resolve()
    if resolved.parent != KAGGLE_ROOT.resolve() or resolved.name != ".validation_workspace":
        raise RuntimeError(f"Unexpected validation cleanup target: {resolved}")
    for attempt in range(6):
        try:
            shutil.rmtree(resolved)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(1 + attempt)


def main() -> None:
    remove_validation_workspace()
    WORKSPACE.mkdir(parents=True)
    shutil.copy2(DATASET_DIR / "TelcoCustomerChurn.csv", WORKSPACE / "TelcoCustomerChurn.csv")
    shutil.copytree(DATASET_DIR / "src", WORKSPACE / "src")

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=1_200,
        kernel_name="python3",
        resources={"metadata": {"path": str(WORKSPACE)}},
    )
    try:
        client.execute()
        executed_code = [cell for cell in notebook.cells if cell.cell_type == "code"]
        errors = [
            output
            for cell in executed_code
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        report = {
            "status": "passed" if not errors else "failed",
            "notebook": NOTEBOOK_PATH.name,
            "cells": len(notebook.cells),
            "code_cells_executed": sum(cell.execution_count is not None for cell in executed_code),
            "error_outputs": len(errors),
        }
        print(json.dumps(report, indent=2))
        if errors:
            raise AssertionError(f"Notebook produced {len(errors)} error outputs.")
    finally:
        remove_validation_workspace()


if __name__ == "__main__":
    main()

"""Shared fixtures for the ``churn_library`` unit tests.

The unit tests never touch the real dataset or the project's output folders:
they use a small synthetic dataset with the same schema as ``bank_data.csv`` and
redirect every output path in ``constants`` to a pytest temporary directory.

Author: Colin Wuest
Date created: 2026-11-19
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

import churn_library as cls
import constants as const

N_ROWS = 240

CATEGORY_VALUES = {
    "Gender": ["M", "F"],
    "Education_Level": ["High School", "Graduate", "Uneducated", "Unknown"],
    "Marital_Status": ["Married", "Single", "Divorced", "Unknown"],
    "Income_Category": ["Less than $40K", "$40K - $60K", "$60K - $80K", "$120K +"],
    "Card_Category": ["Blue", "Silver", "Gold"],
}


@pytest.fixture(name="raw_df")
def fixture_raw_df():
    """Return a synthetic raw dataframe with the schema of ``bank_data.csv``.

    Churn is made to depend on ``Total_Trans_Ct`` so models have signal to learn.

    Returns:
        pandas DataFrame without the ``Churn`` column
    """
    rng = np.random.default_rng(0)
    data = {"CLIENTNUM": np.arange(700_000_000, 700_000_000 + N_ROWS)}
    data.update({col: rng.choice(values, N_ROWS) for col, values in CATEGORY_VALUES.items()})
    data.update({col: rng.normal(50, 10, N_ROWS) for col in const.QUANT_COLUMNS})
    churned = data["Total_Trans_Ct"] < np.quantile(data["Total_Trans_Ct"], 0.3)
    data[const.ATTRITION_COLUMN] = np.where(churned, "Attrited Customer", const.EXISTING_CUSTOMER)
    return pd.DataFrame(data)


@pytest.fixture(name="csv_path")
def fixture_csv_path(raw_df, tmp_path):
    """Write the synthetic data to a csv laid out like ``bank_data.csv`` (index column first).

    Returns:
        path to the csv file
    """
    path = tmp_path / "bank_data.csv"
    raw_df.to_csv(path)
    return str(path)


@pytest.fixture(name="churn_df")
def fixture_churn_df(csv_path):
    """Return the synthetic data as loaded by ``import_data`` (includes ``Churn``).

    Returns:
        pandas DataFrame
    """
    return cls.import_data(csv_path)


@pytest.fixture(name="split_data")
def fixture_split_data(churn_df):
    """Return ``(x_train, x_test, y_train, y_test)`` for the synthetic data.

    Returns:
        tuple of train/test features and labels
    """
    encoded = cls.encoder_helper(churn_df, const.CAT_COLUMNS, const.RESPONSE)
    return cls.perform_feature_engineering(encoded, const.RESPONSE)


@pytest.fixture(name="fitted_rf")
def fixture_fitted_rf(split_data):
    """Return a small random forest fitted on the synthetic training data.

    Returns:
        fitted RandomForestClassifier
    """
    x_train, _, y_train, _ = split_data
    return RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0).fit(
        x_train, y_train
    )


@pytest.fixture(name="output_dirs", autouse=True)
def fixture_output_dirs(tmp_path, monkeypatch):
    """Redirect all output folders and artifact paths to a temporary directory.

    Applied automatically to every unit test so the real ``images/`` and
    ``models/`` folders are never modified.

    Returns:
        dict with the temporary ``eda``, ``results`` and ``models`` directories
    """
    dirs = {name: tmp_path / "out" / name for name in ("eda", "results", "models")}
    monkeypatch.setattr(const, "EDA_DIR", str(dirs["eda"]))
    monkeypatch.setattr(const, "RESULTS_DIR", str(dirs["results"]))
    monkeypatch.setattr(const, "MODELS_DIR", str(dirs["models"]))
    for attr, folder in (
        ("EDA_IMAGES", dirs["eda"]),
        ("RESULT_IMAGES", dirs["results"]),
        ("MODEL_FILES", dirs["models"]),
    ):
        original = getattr(const, attr)
        redirected = {key: str(folder / os.path.basename(pth)) for key, pth in original.items()}
        monkeypatch.setattr(const, attr, redirected)
    return dirs


@pytest.fixture(name="fast_search", autouse=True)
def fixture_fast_search(monkeypatch):
    """Shrink the random forest hyper-parameter search so training takes seconds.

    Returns:
        None
    """
    monkeypatch.setattr(
        const,
        "RF_PARAM_GRID",
        {"n_estimators": [10], "max_depth": [3], "max_features": ["sqrt"]},
    )
    monkeypatch.setattr(const, "RF_SEARCH_ITERATIONS", 1)
    monkeypatch.setattr(const, "RF_SEARCH_CV_FOLDS", 2)


@pytest.fixture(autouse=True)
def close_figures():
    """Close any matplotlib figures a test leaves open.

    Yields:
        None
    """
    yield
    plt.close("all")

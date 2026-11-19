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

    Churn is made to depend on ``Total_Trans_Ct`` (the lowest 30 % churn) so the
    models have a signal to learn. A fixed random seed makes the data reproducible.

    Args:
        None

    Returns:
        raw_df: pandas DataFrame with ``N_ROWS`` rows, the ``CLIENTNUM`` id, the
            ``Attrition_Flag`` target, all ``CAT_COLUMNS`` and ``QUANT_COLUMNS``,
            and no ``Churn`` column
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

    Args:
        raw_df: synthetic raw DataFrame from the ``raw_df`` fixture
        tmp_path: pytest's per-test temporary directory (``pathlib.Path``)

    Returns:
        csv_path: str path of the written ``bank_data.csv`` inside ``tmp_path``
    """
    path = tmp_path / "bank_data.csv"
    raw_df.to_csv(path)
    return str(path)


@pytest.fixture(name="churn_df")
def fixture_churn_df(csv_path):
    """Return the synthetic data as loaded by ``import_data`` (includes ``Churn``).

    Args:
        csv_path: str path of the synthetic csv from the ``csv_path`` fixture

    Returns:
        churn_df: pandas DataFrame with the raw columns plus the binary ``Churn`` column
    """
    return cls.import_data(csv_path)


@pytest.fixture(name="split_data")
def fixture_split_data(churn_df):
    """Return the train/test split of the encoded synthetic data.

    The categorical columns are mean-encoded with ``encoder_helper`` and the data is
    split with ``perform_feature_engineering`` exactly as in the real pipeline.

    Args:
        churn_df: synthetic DataFrame with ``Churn`` from the ``churn_df`` fixture

    Returns:
        split_data: tuple ``(x_train, x_test, y_train, y_test)`` where the ``x_*``
            items are feature DataFrames and the ``y_*`` items are ``Churn`` Series
    """
    encoded = cls.encoder_helper(churn_df, const.CAT_COLUMNS, const.RESPONSE)
    return cls.perform_feature_engineering(encoded, const.RESPONSE)


@pytest.fixture(name="fitted_rf")
def fixture_fitted_rf(split_data):
    """Return a small random forest fitted on the synthetic training data.

    Args:
        split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the
            ``split_data`` fixture; only the training part is used

    Returns:
        fitted_rf: fitted ``RandomForestClassifier`` (10 trees, max depth 3)
    """
    x_train, _, y_train, _ = split_data
    return RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0).fit(
        x_train, y_train
    )


@pytest.fixture(name="output_dirs", autouse=True)
def fixture_output_dirs(tmp_path, monkeypatch):
    """Redirect all output folders and artifact paths to a temporary directory.

    Applied automatically to every unit test so the real ``images/`` and
    ``models/`` folders are never modified. ``EDA_DIR``, ``RESULTS_DIR``,
    ``MODELS_DIR``, ``EDA_IMAGES``, ``RESULT_IMAGES`` and ``MODEL_FILES`` in
    ``constants`` are patched for the duration of the test only.

    Args:
        tmp_path: pytest's per-test temporary directory (``pathlib.Path``)
        monkeypatch: pytest fixture used to patch ``constants`` and undo it afterwards

    Returns:
        output_dirs: dict mapping ``"eda"``, ``"results"`` and ``"models"`` to the
            temporary ``pathlib.Path`` folders (not created until the code under test runs)
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

    Applied automatically to every unit test: ``RF_PARAM_GRID`` is reduced to a
    single small candidate with one search iteration and 2-fold cross-validation.

    Args:
        monkeypatch: pytest fixture used to patch ``constants`` and undo it afterwards

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

    Applied automatically to every unit test; the cleanup runs after the test.

    Args:
        None

    Yields:
        None: control passes to the test; all figures are closed once it finishes
    """
    yield
    plt.close("all")

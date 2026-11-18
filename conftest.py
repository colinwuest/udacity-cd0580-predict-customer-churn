"""Pytest fixtures that inject the ``churn_library`` functions into the tests.

The test functions in ``churn_script_logging_and_tests.py`` take the function
under test as an argument so they can also be called directly from the script.
These fixtures supply those arguments when the tests are collected by pytest.

Author: Colin Wuest
Date created: 2026-11-19
"""

import pytest

import churn_library as cls


@pytest.fixture(name="import_data")
def fixture_import_data():
    """Provide ``churn_library.import_data``."""
    return cls.import_data


@pytest.fixture(name="perform_eda")
def fixture_perform_eda():
    """Provide ``churn_library.perform_eda``."""
    return cls.perform_eda


@pytest.fixture(name="encoder_helper")
def fixture_encoder_helper():
    """Provide ``churn_library.encoder_helper``."""
    return cls.encoder_helper


@pytest.fixture(name="perform_feature_engineering")
def fixture_perform_feature_engineering():
    """Provide ``churn_library.perform_feature_engineering``."""
    return cls.perform_feature_engineering


@pytest.fixture(name="train_models")
def fixture_train_models():
    """Provide ``churn_library.train_models``."""
    return cls.train_models


def pytest_collection_modifyitems(items):
    """Mark the end-to-end tests of the rubric script as ``integration``.

    This allows running only the fast unit tests with ``pytest -m "not integration"``.

    Args:
        items: collected pytest items

    Returns:
        None
    """
    for item in items:
        if item.path.name == "churn_script_logging_and_tests.py":
            item.add_marker(pytest.mark.integration)

"""Unit tests and logging for the functions in ``churn_library.py``.

Each test calls one library function, asserts that its outputs are not empty
and that the expected files (images, models) were created, and logs the result
as ``SUCCESS`` (INFO) or ``ERROR`` to ``./logs/churn_library.log``.

Run as a script (tests run in pipeline order, log file is recreated)::

    uv run python churn_script_logging_and_tests.py

or with pytest (functions are injected via the fixtures in ``conftest.py``)::

    uv run pytest

Author: Colin Wuest
Date created: 2026-11-19
"""

import logging
import os

import joblib

import churn_library as cls
import constants as const

LOGS_DIR = const.LOGS_DIR
LOG_FILE = const.LOG_FILE
DATA_PATH = const.DATA_PATH

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_logging(log_file=LOG_FILE, mode="a"):
    """Attach a file handler writing INFO and ERROR messages to ``log_file``.

    A dedicated handler is added to the root logger (instead of
    ``logging.basicConfig``) so that logging also works under pytest, which
    installs its own handlers. Messages from ``churn_library`` are captured too.

    Args:
        log_file: path of the .log file to write
        mode: file mode, ``"w"`` to start a fresh log or ``"a"`` to append

    Returns:
        None
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    root = logging.getLogger()
    target = os.path.abspath(log_file)
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == target:
            return
    file_handler = logging.FileHandler(log_file, mode=mode, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(file_handler)
    root.setLevel(logging.INFO)


configure_logging()
logger = logging.getLogger("churn_tests")


def _remove_files(paths):
    """Delete existing artifacts so a test proves they are (re)created.

    Args:
        paths: iterable of file paths

    Returns:
        None
    """
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def _assert_files_exist(paths):
    """Assert that every path exists and is a non-empty file.

    Args:
        paths: iterable of file paths

    Returns:
        None
    """
    for path in paths:
        assert os.path.isfile(path), f"missing file {path}"
        assert os.path.getsize(path) > 0, f"empty file {path}"


def _prepared_data():
    """Import the data and encode the categorical columns.

    Args:
        None

    Returns:
        df: pandas DataFrame ready for feature engineering
    """
    df = cls.import_data(DATA_PATH)
    return cls.encoder_helper(df, const.CAT_COLUMNS, const.RESPONSE)


def test_import(import_data):
    """Test data import: the file loads and the dataframe has rows, columns and the response.

    Args:
        import_data: function under test

    Returns:
        None
    """
    logger.info("Running test_import")
    try:
        df = import_data(DATA_PATH)
        logger.info("test_import: file %s loaded", DATA_PATH)
    except FileNotFoundError as err:
        logger.error("ERROR test_import: the file %s wasn't found", DATA_PATH)
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        assert const.RESPONSE in df.columns
        assert set(df[const.RESPONSE].unique()) <= {0, 1}
        logger.info("SUCCESS test_import: dataframe shape %s", df.shape)
    except AssertionError as err:
        logger.error(
            "ERROR test_import: the file doesn't appear to have rows/columns "
            "or the '%s' column is missing",
            const.RESPONSE,
        )
        raise err


def test_eda(perform_eda):
    """Test EDA: every expected EDA image is written to the EDA folder.

    Args:
        perform_eda: function under test

    Returns:
        None
    """
    logger.info("Running test_eda")
    try:
        df = cls.import_data(DATA_PATH)
        _remove_files(const.EDA_IMAGES.values())
        perform_eda(df)
        _assert_files_exist(const.EDA_IMAGES.values())
        logger.info("SUCCESS test_eda: %d EDA images saved", len(const.EDA_IMAGES))
    except AssertionError as err:
        logger.error("ERROR test_eda: EDA image not created - %s", err)
        raise err
    except Exception as err:
        logger.error("ERROR test_eda: perform_eda raised %s: %s", type(err).__name__, err)
        raise err


def test_encoder_helper(encoder_helper):
    """Test encoding: one mean-response column per category, values are churn rates.

    Args:
        encoder_helper: function under test

    Returns:
        None
    """
    logger.info("Running test_encoder_helper")
    try:
        df = cls.import_data(DATA_PATH)
        encoded = encoder_helper(df, const.CAT_COLUMNS, const.RESPONSE)

        assert encoded.shape[0] == df.shape[0]
        for column in const.CAT_COLUMNS:
            new_column = f"{column}_{const.RESPONSE}"
            assert new_column in encoded.columns, f"missing column {new_column}"
            assert encoded[new_column].notna().all(), f"NaN values in {new_column}"
            assert encoded[new_column].between(0, 1).all(), f"{new_column} out of [0, 1]"
        logger.info(
            "SUCCESS test_encoder_helper: encoded %d categorical columns",
            len(const.CAT_COLUMNS),
        )
    except AssertionError as err:
        logger.error("ERROR test_encoder_helper: %s", err)
        raise err
    except Exception as err:
        logger.error(
            "ERROR test_encoder_helper: encoder_helper raised %s: %s", type(err).__name__, err
        )
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """Test feature engineering: non-empty, consistent train/test splits with all features.

    Args:
        perform_feature_engineering: function under test

    Returns:
        None
    """
    logger.info("Running test_perform_feature_engineering")
    try:
        df = _prepared_data()
        x_train, x_test, y_train, y_test = perform_feature_engineering(df, const.RESPONSE)

        assert len(x_train) > 0 and len(x_test) > 0
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        assert len(x_train) + len(x_test) == len(df)
        assert list(x_train.columns) == cls.get_feature_columns(const.RESPONSE)
        logger.info(
            "SUCCESS test_perform_feature_engineering: train %s, test %s",
            x_train.shape,
            x_test.shape,
        )
    except AssertionError as err:
        logger.error("ERROR test_perform_feature_engineering: invalid split - %s", err)
        raise err
    except Exception as err:
        logger.error(
            "ERROR test_perform_feature_engineering: raised %s: %s", type(err).__name__, err
        )
        raise err


def test_train_models(train_models):
    """Test training: both models and all result images are saved and models load again.

    Args:
        train_models: function under test

    Returns:
        None
    """
    logger.info("Running test_train_models (this can take a few minutes)")
    try:
        df = _prepared_data()
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(df, const.RESPONSE)
        artifacts = list(const.MODEL_FILES.values()) + list(const.RESULT_IMAGES.values())
        _remove_files(artifacts)

        train_models(x_train, x_test, y_train, y_test)

        _assert_files_exist(artifacts)
        for model_path in const.MODEL_FILES.values():
            model = joblib.load(model_path)
            assert len(model.predict(x_test)) == len(x_test), f"{model_path} cannot predict"
        logger.info(
            "SUCCESS test_train_models: %d models and %d result images saved",
            len(const.MODEL_FILES),
            len(const.RESULT_IMAGES),
        )
    except AssertionError as err:
        logger.error("ERROR test_train_models: artifact check failed - %s", err)
        raise err
    except Exception as err:
        logger.error("ERROR test_train_models: raised %s: %s", type(err).__name__, err)
        raise err


def main():
    """Run every test in pipeline order, writing a fresh log file.

    Args:
        None

    Returns:
        None
    """
    logging.getLogger().handlers.clear()
    configure_logging(mode="w")
    logger.info("Starting churn_library test run")

    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)

    logger.info("All tests passed")
    print(f"Tests completed. Check {LOG_FILE} for details.")


if __name__ == "__main__":
    main()

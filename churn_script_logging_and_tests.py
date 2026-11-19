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
# tolerance for comparing an encoded value with the training churn rate
FLOAT_TOLERANCE = 1e-9
# a stratified split keeps train and test churn rates within this gap
MAX_CHURN_RATE_GAP = 0.01


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
    """Test encoding: fitted on the training split, it encodes unseen data without leakage.

    The encoder is fitted on the training rows only and applied to the test rows:
    every categorical column must become a churn rate between 0 and 1, the numeric
    columns must pass through, and an unseen category must fall back to the overall
    training churn rate.

    Args:
        encoder_helper: function under test

    Returns:
        None
    """
    logger.info("Running test_encoder_helper")
    try:
        df = cls.import_data(DATA_PATH)
        x_train, x_test, y_train, _ = cls.perform_feature_engineering(df, const.RESPONSE)
        encoder = encoder_helper(const.CAT_COLUMNS).fit(x_train, y_train)
        encoded = encoder.transform(x_test)

        assert encoded.shape[0] == x_test.shape[0]
        assert list(encoded.columns) == const.CAT_COLUMNS + const.QUANT_COLUMNS
        for column in const.CAT_COLUMNS:
            assert encoded[column].notna().all(), f"NaN values in {column}"
            assert encoded[column].between(0, 1).all(), f"{column} out of [0, 1]"

        unseen = x_test.head(1).copy()
        unseen["Card_Category"] = "Never seen before"
        fallback = encoder.transform(unseen)["Card_Category"].iloc[0]
        assert abs(fallback - y_train.mean()) < FLOAT_TOLERANCE, (
            "unseen category not mapped to churn rate"
        )
        logger.info(
            "SUCCESS test_encoder_helper: fitted on %d training rows, encoded %d "
            "categorical columns of %d test rows; unseen categories -> %.3f",
            len(x_train),
            len(const.CAT_COLUMNS),
            len(x_test),
            fallback,
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
    """Test feature engineering: raw, non-overlapping, stratified train/test splits.

    Args:
        perform_feature_engineering: function under test

    Returns:
        None
    """
    logger.info("Running test_perform_feature_engineering")
    try:
        df = cls.import_data(DATA_PATH)
        x_train, x_test, y_train, y_test = perform_feature_engineering(df, const.RESPONSE)

        assert len(x_train) > 0 and len(x_test) > 0
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        assert len(x_train) + len(x_test) == len(df)
        assert list(x_train.columns) == cls.get_feature_columns()
        assert const.RESPONSE not in x_train.columns, "response leaked into the features"
        assert set(x_train.index).isdisjoint(x_test.index), "rows in both train and test"
        assert abs(y_train.mean() - y_test.mean()) < MAX_CHURN_RATE_GAP, "split is not stratified"
        assert x_train[const.CAT_COLUMNS].map(lambda val: isinstance(val, str)).all().all(), (
            "categorical columns were encoded before the split"
        )
        logger.info(
            "SUCCESS test_perform_feature_engineering: train %s, test %s, churn rate %.3f / %.3f",
            x_train.shape,
            x_test.shape,
            y_train.mean(),
            y_test.mean(),
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
    """Test training: models and result images are saved, and models predict from raw rows.

    Args:
        train_models: function under test

    Returns:
        None
    """
    logger.info("Running test_train_models (this can take a few minutes)")
    try:
        df = cls.import_data(DATA_PATH)
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(df, const.RESPONSE)
        artifacts = list(const.MODEL_FILES.values()) + list(const.RESULT_IMAGES.values())
        _remove_files(artifacts)

        train_models(x_train, x_test, y_train, y_test)

        _assert_files_exist(artifacts)
        raw_rows = df.loc[x_test.index]  # full raw csv rows: ids, target, string categories
        for model_path in const.MODEL_FILES.values():
            model = joblib.load(model_path)
            probabilities = model.predict_proba(raw_rows)[:, 1]
            assert len(probabilities) == len(raw_rows), f"{model_path} cannot predict"
            assert ((probabilities >= 0) & (probabilities <= 1)).all()
        logger.info(
            "SUCCESS test_train_models: %d model pipelines and %d result images saved; "
            "models predict from raw customer rows",
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

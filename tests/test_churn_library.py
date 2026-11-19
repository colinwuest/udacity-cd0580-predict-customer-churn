"""Unit tests for ``churn_library``.

Each library function is tested on its own with small synthetic data (see
``tests/conftest.py``). All outputs go to a temporary directory and the
random forest search is shrunk, so the suite runs in seconds. ``TestNoLeakage``
checks that no test-set information reaches the fitted models::

    uv run pytest tests

Author: Colin Wuest
Date created: 2026-11-19
"""

# Test classes only group related tests, so a single public method is fine.
# pylint: disable=too-few-public-methods

import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

import churn_library as cls
import constants as const


def _assert_image(path):
    """Assert that ``path`` is a non-empty PNG file.

    Args:
        path: str or ``pathlib.Path`` of the image file to check

    Returns:
        None (raises ``AssertionError`` if the file is missing or not a PNG)
    """
    assert os.path.isfile(path), f"missing {path}"
    with open(path, "rb") as file:
        assert file.read(8) == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"


def _fitted_logistic(x_train, y_train):
    """Fit the project's logistic regression pipeline on training data.

    Args:
        x_train: DataFrame of raw training features
        y_train: Series of training responses

    Returns:
        lrc: fitted ``encode -> StandardScaler -> LogisticRegression`` pipeline
    """
    _, lrc = cls.build_models()
    return lrc.fit(x_train, y_train)


class TestCreateOutputDirectories:
    """Tests for ``create_output_directories``."""

    def test_creates_all_directories(self, output_dirs):
        """All three output folders exist afterwards.

        Args:
            output_dirs: dict of temporary output folders (``eda``, ``results``, ``models``)
                from the ``output_dirs`` fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        cls.create_output_directories()
        for folder in output_dirs.values():
            assert folder.is_dir()

    def test_is_idempotent(self, output_dirs):
        """Calling it twice does not raise.

        Args:
            output_dirs: dict of temporary output folders (``eda``, ``results``, ``models``)
                from the ``output_dirs`` fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        cls.create_output_directories()
        cls.create_output_directories()
        assert output_dirs["models"].is_dir()


class TestImportData:
    """Tests for ``import_data``."""

    def test_returns_all_rows(self, csv_path, raw_df):
        """Every csv row is loaded.

        Args:
            csv_path: str path of the synthetic csv from the ``csv_path`` fixture
            raw_df: synthetic raw DataFrame from the ``raw_df`` fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        df = cls.import_data(csv_path)
        assert len(df) == len(raw_df)

    def test_index_column_is_not_a_feature(self, csv_path):
        """The csv index column is used as index, not loaded as ``Unnamed: 0``.

        Args:
            csv_path: str path of the synthetic csv from the ``csv_path`` fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        df = cls.import_data(csv_path)
        assert not any(col.startswith("Unnamed") for col in df.columns)

    def test_adds_binary_churn_column(self, csv_path):
        """``Churn`` is 1 exactly for attrited customers and 0 otherwise.

        Args:
            csv_path: str path of the synthetic csv from the ``csv_path`` fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        df = cls.import_data(csv_path)
        expected = (df[const.ATTRITION_COLUMN] != const.EXISTING_CUSTOMER).astype(int)
        pd.testing.assert_series_equal(df[const.RESPONSE], expected, check_names=False)
        assert set(df[const.RESPONSE].unique()) == {0, 1}

    def test_missing_file_raises(self, tmp_path):
        """A missing file raises ``FileNotFoundError``.

        Args:
            tmp_path: pytest's per-test temporary directory (``pathlib.Path``)

        Returns:
            None (the test fails if ``FileNotFoundError`` is not raised)
        """
        with pytest.raises(FileNotFoundError):
            cls.import_data(str(tmp_path / "does_not_exist.csv"))

    def test_logs_import(self, csv_path, caplog):
        """A successful import is logged at INFO level.

        Args:
            csv_path: str path of the synthetic csv from the ``csv_path`` fixture
            caplog: pytest fixture that captures log records emitted during the test

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        with caplog.at_level(logging.INFO, logger="churn_library"):
            cls.import_data(csv_path)
        assert "Imported" in caplog.text


class TestPerformEda:
    """Tests for ``perform_eda``."""

    def test_saves_every_eda_image(self, churn_df):
        """All configured EDA images are written as PNG files.

        Args:
            churn_df: synthetic DataFrame with ``Churn`` from the ``churn_df`` fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        cls.perform_eda(churn_df)
        for path in const.EDA_IMAGES.values():
            _assert_image(path)

    def test_closes_figures(self, churn_df):
        """No matplotlib figures stay open (avoids memory leaks in long runs).

        Args:
            churn_df: synthetic DataFrame with ``Churn`` from the ``churn_df`` fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        plt.close("all")
        cls.perform_eda(churn_df)
        assert not plt.get_fignums()

    def test_does_not_modify_input(self, churn_df):
        """The dataframe passed in is left unchanged.

        Args:
            churn_df: synthetic DataFrame with ``Churn`` from the ``churn_df`` fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        before = churn_df.copy()
        cls.perform_eda(churn_df)
        pd.testing.assert_frame_equal(churn_df, before)

    def test_missing_column_raises(self, churn_df):
        """A dataframe without a plotted column raises ``KeyError``.

        Args:
            churn_df: synthetic DataFrame with ``Churn`` from the ``churn_df`` fixture

        Returns:
            None (the test fails if ``KeyError`` is not raised)
        """
        with pytest.raises(KeyError):
            cls.perform_eda(churn_df.drop(columns=["Customer_Age"]))


class TestEncoderHelper:
    """Tests for ``encoder_helper`` (the mean-response encoder used inside the pipelines)."""

    @pytest.fixture(name="small_data")
    def fixture_small_data(self, monkeypatch):
        """Return hand-made data with known churn rates and disable smoothing.

        The six base rows are repeated four times so the encoder's 2-fold
        cross-fitting has enough rows per class. ``TARGET_ENCODER_SMOOTH`` is set to
        0 so the learned encodings are the plain category churn rates.

        Args:
            monkeypatch: pytest fixture used to patch the encoder settings in ``constants``

        Returns:
            small_data: tuple ``(x_small, y_small)``; ``x_small`` has ``Gender``,
                ``Card_Category`` and a numeric ``Score`` column, ``y_small`` is the
                churn Series. Churn rates are M 0.5, F 0.25, Blue 2/3, Gold 0
        """
        monkeypatch.setattr(const, "TARGET_ENCODER_SMOOTH", 0.0)
        monkeypatch.setattr(const, "TARGET_ENCODER_CV_FOLDS", 2)
        x_small = pd.DataFrame(
            {
                "Gender": ["M", "M", "F", "F", "F", "F"] * 4,
                "Card_Category": ["Blue", "Gold", "Blue", "Blue", "Gold", "Gold"] * 4,
                "Score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 4,
            }
        )
        y_small = pd.Series([1, 0, 1, 0, 0, 0] * 4, name="Churn")
        return x_small, y_small

    def test_returns_unfitted_column_transformer(self):
        """The helper builds a fresh, unfitted ``ColumnTransformer``.

        Args:
            None

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        encoder = cls.encoder_helper(const.CAT_COLUMNS)
        assert isinstance(encoder, ColumnTransformer)
        assert not hasattr(encoder, "transformers_")

    def test_learns_category_churn_rates(self, small_data):
        """After fitting, each category is mapped to its training churn rate.

        Args:
            small_data: tuple ``(x_small, y_small)`` with known churn rates from the
                ``small_data`` fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        x_small, y_small = small_data
        encoder = cls.encoder_helper(["Gender", "Card_Category"], ["Score"])
        encoded = encoder.fit(x_small, y_small).transform(x_small.head(6))
        assert encoded["Gender"].tolist() == pytest.approx([0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
        assert encoded["Card_Category"].tolist() == pytest.approx(
            [2 / 3, 0.0, 2 / 3, 2 / 3, 0.0, 0.0]
        )

    def test_output_columns_and_dropped_extras(self, churn_df):
        """Output is the encoded categories then the numeric columns; ids and target are dropped.

        Args:
            churn_df: synthetic DataFrame with ``Churn`` from the ``churn_df`` fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        encoder = cls.encoder_helper(const.CAT_COLUMNS)
        encoded = encoder.fit_transform(churn_df, churn_df[const.RESPONSE])
        assert list(encoded.columns) == const.CAT_COLUMNS + const.QUANT_COLUMNS
        for dropped in ("CLIENTNUM", const.ATTRITION_COLUMN, const.RESPONSE):
            assert dropped not in encoded.columns

    def test_numeric_columns_pass_through(self, split_data):
        """Numeric columns are passed through unchanged.

        Args:
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        x_train, x_test, y_train, _ = split_data
        encoded = cls.encoder_helper(const.CAT_COLUMNS).fit(x_train, y_train).transform(x_test)
        pd.testing.assert_frame_equal(encoded[const.QUANT_COLUMNS], x_test[const.QUANT_COLUMNS])

    def test_unseen_category_gets_overall_rate(self, small_data):
        """A category never seen in training is encoded as the overall training churn rate.

        Args:
            small_data: tuple ``(x_small, y_small)`` with known churn rates from the
                ``small_data`` fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        x_small, y_small = small_data
        encoder = cls.encoder_helper(["Gender", "Card_Category"], ["Score"]).fit(x_small, y_small)
        unseen = pd.DataFrame({"Gender": ["X"], "Card_Category": ["Diamond"], "Score": [0.0]})
        encoded = encoder.transform(unseen)
        assert encoded["Gender"].iloc[0] == pytest.approx(y_small.mean())
        assert encoded["Card_Category"].iloc[0] == pytest.approx(y_small.mean())

    def test_training_rows_use_out_of_fold_encodings(self, split_data):
        """``fit_transform`` cross-fits, so training rows do not see their own label.

        Args:
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        x_train, _, y_train, _ = split_data
        encoder = cls.encoder_helper(const.CAT_COLUMNS)
        out_of_fold = encoder.fit_transform(x_train, y_train)
        in_sample = encoder.transform(x_train)
        assert not np.allclose(out_of_fold[const.CAT_COLUMNS], in_sample[const.CAT_COLUMNS])

    def test_encoded_values_are_rates(self, split_data):
        """Encoded categorical values are churn rates between 0 and 1 without NaNs.

        Args:
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        x_train, x_test, y_train, _ = split_data
        encoded = cls.encoder_helper(const.CAT_COLUMNS).fit(x_train, y_train).transform(x_test)
        for column in const.CAT_COLUMNS:
            assert encoded[column].notna().all()
            assert encoded[column].between(0, 1).all()

    def test_does_not_modify_input(self, split_data):
        """Fitting and transforming leave the input dataframe unchanged.

        Args:
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        x_train, _, y_train, _ = split_data
        before = x_train.copy()
        cls.encoder_helper(const.CAT_COLUMNS).fit_transform(x_train, y_train)
        pd.testing.assert_frame_equal(x_train, before)

    def test_unknown_column_raises(self, small_data):
        """Encoding a column that does not exist raises ``ValueError``.

        Args:
            small_data: tuple ``(x_small, y_small)`` with known churn rates from the
                ``small_data`` fixture

        Returns:
            None (the test fails if ``ValueError`` is not raised)
        """
        x_small, y_small = small_data
        with pytest.raises(ValueError):
            cls.encoder_helper(["Not_A_Column"], ["Score"]).fit(x_small, y_small)


class TestGetFeatureColumns:
    """Tests for ``get_feature_columns``."""

    def test_quantitative_then_categorical(self):
        """The raw quantitative columns come first, then the raw categorical columns.

        Args:
            None

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        assert cls.get_feature_columns() == const.QUANT_COLUMNS + const.CAT_COLUMNS

    def test_has_no_duplicates_or_target(self):
        """No feature appears twice and neither the target nor the id is a feature.

        Args:
            None

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        columns = cls.get_feature_columns()
        assert len(columns) == len(set(columns))
        for excluded in (const.RESPONSE, const.ATTRITION_COLUMN, "CLIENTNUM"):
            assert excluded not in columns


class TestPerformFeatureEngineering:
    """Tests for ``perform_feature_engineering``."""

    def test_split_sizes(self, split_data, churn_df):
        """Train/test sizes follow ``TEST_SIZE`` and cover every row.

        Args:
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture
            churn_df: synthetic DataFrame with ``Churn`` from the ``churn_df`` fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        x_train, x_test, y_train, y_test = split_data
        assert len(x_train) + len(x_test) == len(churn_df)
        assert len(x_test) == pytest.approx(len(churn_df) * const.TEST_SIZE, abs=1)
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)

    def test_features_are_raw(self, split_data):
        """X holds the raw feature columns; categories are still unencoded strings.

        Args:
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        x_train, x_test, _, _ = split_data
        assert list(x_train.columns) == cls.get_feature_columns()
        assert list(x_test.columns) == list(x_train.columns)
        assert const.RESPONSE not in x_train.columns
        for column in const.CAT_COLUMNS:
            assert x_train[column].map(lambda val: isinstance(val, str)).all()

    def test_no_overlap_between_train_and_test(self, split_data):
        """No row ends up in both train and test sets.

        Args:
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        x_train, x_test, _, _ = split_data
        assert set(x_train.index).isdisjoint(x_test.index)

    def test_split_is_stratified(self, split_data, churn_df):
        """Train and test sets keep the overall churn rate (stratified split).

        Args:
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture
            churn_df: synthetic DataFrame with ``Churn`` from the ``churn_df`` fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        _, _, y_train, y_test = split_data
        overall = churn_df[const.RESPONSE].mean()
        assert y_train.mean() == pytest.approx(overall, abs=0.01)
        assert y_test.mean() == pytest.approx(overall, abs=0.02)

    def test_is_reproducible(self, churn_df):
        """The split is deterministic thanks to the fixed random state.

        Args:
            churn_df: synthetic DataFrame with ``Churn`` from the ``churn_df`` fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        first = cls.perform_feature_engineering(churn_df, const.RESPONSE)
        second = cls.perform_feature_engineering(churn_df, const.RESPONSE)
        assert first[0].index.equals(second[0].index)


class TestBuildModels:
    """Tests for ``build_models``."""

    def test_pipelines_start_with_encoder(self):
        """Both models are unfitted pipelines whose first step is the encoder.

        Args:
            None

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        cv_rfc, lrc = cls.build_models()
        assert isinstance(cv_rfc, RandomizedSearchCV)
        for pipeline in (cv_rfc.estimator, lrc):
            assert isinstance(pipeline, Pipeline)
            assert pipeline.steps[0][0] == "encode"
            assert isinstance(pipeline.steps[0][1], ColumnTransformer)

    def test_search_tunes_model_step(self):
        """The search parameters target the model step of the pipeline.

        Args:
            None

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        cv_rfc, _ = cls.build_models()
        assert all(key.startswith("model__") for key in cv_rfc.param_distributions)


class TestClassificationReportImage:
    """Tests for ``classification_report_image``."""

    def test_saves_both_reports(self, split_data):
        """One report image per model is written.

        Args:
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        _, _, y_train, y_test = split_data
        cls.classification_report_image(y_train, y_test, y_train, y_train, y_test, y_test)
        _assert_image(const.RESULT_IMAGES["rf_report"])
        _assert_image(const.RESULT_IMAGES["lr_report"])

    def test_mismatched_lengths_raise(self, split_data):
        """Predictions with the wrong length raise ``ValueError``.

        Args:
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture

        Returns:
            None (the test fails if ``ValueError`` is not raised)
        """
        _, _, y_train, y_test = split_data
        with pytest.raises(ValueError):
            cls.classification_report_image(y_train, y_test, y_train[:-1], y_train, y_test, y_test)


class TestFeatureImportancePlot:
    """Tests for ``feature_importance_plot``."""

    def test_saves_plot_for_pipeline(self, fitted_rf, split_data, tmp_path):
        """A fitted pipeline with raw rows is accepted and the plot is saved.

        Args:
            fitted_rf: fitted ``encode -> random forest`` pipeline from the ``fitted_rf``
                fixture
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture
            tmp_path: pytest's per-test temporary directory (``pathlib.Path``)

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        output = tmp_path / "importance.png"
        cls.feature_importance_plot(fitted_rf, split_data[1], str(output))
        _assert_image(output)

    def test_saves_plot_for_bare_estimator(self, fitted_rf, split_data, tmp_path):
        """A bare forest with already-encoded rows is accepted too.

        Args:
            fitted_rf: fitted ``encode -> random forest`` pipeline from the ``fitted_rf``
                fixture
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture
            tmp_path: pytest's per-test temporary directory (``pathlib.Path``)

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        output = tmp_path / "nested" / "dir" / "importance.png"
        encoded = fitted_rf[:-1].transform(split_data[1])
        cls.feature_importance_plot(fitted_rf[-1], encoded, str(output))
        _assert_image(output)

    def test_model_without_importances_raises(self, split_data, tmp_path):
        """Models without ``feature_importances_`` raise ``AttributeError``.

        Args:
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture
            tmp_path: pytest's per-test temporary directory (``pathlib.Path``)

        Returns:
            None (the test fails if ``AttributeError`` is not raised)
        """
        x_train, _, y_train, _ = split_data
        model = _fitted_logistic(x_train, y_train)
        with pytest.raises(AttributeError):
            cls.feature_importance_plot(model, x_train, str(tmp_path / "x.png"))


class TestShapSummaryPlot:
    """Tests for ``shap_summary_plot``."""

    def test_saves_plot(self, fitted_rf, split_data, tmp_path):
        """The SHAP summary of a fitted pipeline is written to the requested path.

        Args:
            fitted_rf: fitted ``encode -> random forest`` pipeline from the ``fitted_rf``
                fixture
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture
            tmp_path: pytest's per-test temporary directory (``pathlib.Path``)

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        output = tmp_path / "shap.png"
        cls.shap_summary_plot(fitted_rf, split_data[1].head(30), str(output))
        _assert_image(output)


class TestRocCurvePlot:
    """Tests for ``roc_curve_plot``."""

    def test_saves_plot_with_named_curves(self, fitted_rf, split_data, tmp_path):
        """A labelled curve is drawn per model pipeline and the plot is saved.

        Args:
            fitted_rf: fitted ``encode -> random forest`` pipeline from the ``fitted_rf``
                fixture
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture
            tmp_path: pytest's per-test temporary directory (``pathlib.Path``)

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        x_train, x_test, y_train, y_test = split_data
        models = {
            "Random Forest": fitted_rf,
            "Logistic Regression": _fitted_logistic(x_train, y_train),
        }
        output = tmp_path / "roc.png"

        cls.roc_curve_plot(models, x_test, y_test, str(output))

        _assert_image(output)


class TestTrainModels:
    """Tests for ``train_models`` (with the shrunken hyper-parameter search)."""

    def test_saves_models_and_images(self, split_data):
        """Both model pipelines and every result image are written.

        Args:
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        cls.train_models(*split_data)
        for path in const.MODEL_FILES.values():
            assert os.path.getsize(path) > 0
        for path in const.RESULT_IMAGES.values():
            _assert_image(path)

    def test_saved_models_predict_from_raw_rows(self, split_data, churn_df):
        """Saved pipelines predict from raw csv rows, including ids and unseen categories.

        Args:
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture
            churn_df: synthetic DataFrame with ``Churn`` from the ``churn_df`` fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        cls.train_models(*split_data)
        raw_rows = churn_df.loc[split_data[1].index].copy()
        raw_rows.loc[raw_rows.index[0], "Card_Category"] = "Diamond"
        for path in const.MODEL_FILES.values():
            model = joblib.load(path)
            assert isinstance(model, Pipeline)
            predictions = model.predict(raw_rows)
            assert len(predictions) == len(raw_rows)
            assert set(predictions) <= {0, 1}

    def test_random_forest_beats_chance(self, split_data):
        """The tuned random forest learns the synthetic signal (accuracy well above chance).

        Args:
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        _, x_test, _, y_test = split_data
        cls.train_models(*split_data)
        model = joblib.load(const.MODEL_FILES["rfc"])
        assert model.score(x_test, y_test) > 0.8

    def test_logs_test_scores(self, split_data, caplog):
        """Test-set ROC AUC of both models is logged.

        Args:
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture
            caplog: pytest fixture that captures log records emitted during the test

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        with caplog.at_level(logging.INFO, logger="churn_library"):
            cls.train_models(*split_data)
        assert caplog.text.count("Test ROC AUC") == 2


class TestNoLeakage:
    """Checks that no test-set information reaches the fitted models."""

    def test_encoder_learned_from_training_rows_only(self, split_data):
        """The saved model's category churn rates equal those learned from the training split.

        Args:
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        x_train, x_test, y_train, y_test = split_data
        cls.train_models(x_train, x_test, y_train, y_test)
        saved = joblib.load(const.MODEL_FILES["rfc"])[0].named_transformers_["mean_response"]

        train_only = cls.encoder_helper(const.CAT_COLUMNS).fit(x_train, y_train)
        expected = train_only.named_transformers_["mean_response"]
        all_rows = cls.encoder_helper(const.CAT_COLUMNS).fit(
            pd.concat([x_train, x_test]), pd.concat([y_train, y_test])
        )
        leaky = all_rows.named_transformers_["mean_response"]

        encodings = zip(saved.encodings_, expected.encodings_, leaky.encodings_, strict=True)
        for got, want, leak in encodings:
            np.testing.assert_allclose(got, want)
            assert not np.allclose(got, leak)
        assert saved.target_mean_ == pytest.approx(y_train.mean())

    def test_test_labels_do_not_change_models(self, split_data):
        """Training with different test labels yields identical models and predictions.

        Args:
            split_data: tuple ``(x_train, x_test, y_train, y_test)`` from the ``split_data``
                fixture

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        x_train, x_test, y_train, y_test = split_data
        cls.train_models(x_train, x_test, y_train, y_test)
        original = {key: joblib.load(path) for key, path in const.MODEL_FILES.items()}

        flipped = 1 - y_test
        cls.train_models(x_train, x_test, y_train, flipped)
        for key, path in const.MODEL_FILES.items():
            retrained = joblib.load(path)
            np.testing.assert_array_equal(
                original[key].predict_proba(x_test), retrained.predict_proba(x_test)
            )


class TestMain:
    """Tests for ``main`` (the full pipeline on synthetic data)."""

    def test_runs_end_to_end(self, csv_path, monkeypatch):
        """The pipeline produces every EDA image, result image and model.

        Args:
            csv_path: str path of the synthetic csv from the ``csv_path`` fixture
            monkeypatch: pytest fixture used to point ``constants.DATA_PATH`` at the synthetic csv

        Returns:
            None (the test fails with an ``AssertionError`` if a check does not hold)
        """
        monkeypatch.setattr(const, "DATA_PATH", csv_path)
        cls.main()
        artifacts = (
            list(const.EDA_IMAGES.values())
            + list(const.RESULT_IMAGES.values())
            + list(const.MODEL_FILES.values())
        )
        for path in artifacts:
            assert os.path.isfile(path), f"missing {path}"

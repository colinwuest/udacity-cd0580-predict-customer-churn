"""Unit tests for ``churn_library``.

Each library function is tested on its own with small synthetic data (see
``tests/conftest.py``). All outputs go to a temporary directory and the
random forest search is shrunk, so the suite runs in seconds::

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
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import churn_library as cls
import constants as const


def _assert_image(path):
    """Assert that ``path`` is a non-empty PNG file.

    Args:
        path: file path

    Returns:
        None
    """
    assert os.path.isfile(path), f"missing {path}"
    with open(path, "rb") as file:
        assert file.read(8) == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"


class TestCreateOutputDirectories:
    """Tests for ``create_output_directories``."""

    def test_creates_all_directories(self, output_dirs):
        """All three output folders exist afterwards."""
        cls.create_output_directories()
        for folder in output_dirs.values():
            assert folder.is_dir()

    def test_is_idempotent(self, output_dirs):
        """Calling it twice does not raise."""
        cls.create_output_directories()
        cls.create_output_directories()
        assert output_dirs["models"].is_dir()


class TestImportData:
    """Tests for ``import_data``."""

    def test_returns_all_rows(self, csv_path, raw_df):
        """Every csv row is loaded."""
        df = cls.import_data(csv_path)
        assert len(df) == len(raw_df)

    def test_index_column_is_not_a_feature(self, csv_path):
        """The csv index column is used as index, not loaded as ``Unnamed: 0``."""
        df = cls.import_data(csv_path)
        assert not any(col.startswith("Unnamed") for col in df.columns)

    def test_adds_binary_churn_column(self, csv_path):
        """``Churn`` is 1 exactly for attrited customers and 0 otherwise."""
        df = cls.import_data(csv_path)
        expected = (df[const.ATTRITION_COLUMN] != const.EXISTING_CUSTOMER).astype(int)
        pd.testing.assert_series_equal(df[const.RESPONSE], expected, check_names=False)
        assert set(df[const.RESPONSE].unique()) == {0, 1}

    def test_missing_file_raises(self, tmp_path):
        """A missing file raises ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            cls.import_data(str(tmp_path / "does_not_exist.csv"))

    def test_logs_import(self, csv_path, caplog):
        """A successful import is logged at INFO level."""
        with caplog.at_level(logging.INFO, logger="churn_library"):
            cls.import_data(csv_path)
        assert "Imported" in caplog.text


class TestPerformEda:
    """Tests for ``perform_eda``."""

    def test_saves_every_eda_image(self, churn_df):
        """All configured EDA images are written as PNG files."""
        cls.perform_eda(churn_df)
        for path in const.EDA_IMAGES.values():
            _assert_image(path)

    def test_closes_figures(self, churn_df):
        """No matplotlib figures stay open (avoids memory leaks in long runs)."""
        plt.close("all")
        cls.perform_eda(churn_df)
        assert not plt.get_fignums()

    def test_does_not_modify_input(self, churn_df):
        """The dataframe passed in is left unchanged."""
        before = churn_df.copy()
        cls.perform_eda(churn_df)
        pd.testing.assert_frame_equal(churn_df, before)

    def test_missing_column_raises(self, churn_df):
        """A dataframe without a plotted column raises ``KeyError``."""
        with pytest.raises(KeyError):
            cls.perform_eda(churn_df.drop(columns=["Customer_Age"]))


class TestEncoderHelper:
    """Tests for ``encoder_helper``."""

    @pytest.fixture(name="small_df")
    def fixture_small_df(self):
        """Return a hand-made dataframe with known churn rates per category."""
        return pd.DataFrame(
            {
                "Gender": ["M", "M", "F", "F", "F", "F"],
                "Card_Category": ["Blue", "Gold", "Blue", "Blue", "Gold", "Gold"],
                "Churn": [1, 0, 1, 0, 0, 0],
            }
        )

    def test_computes_mean_response(self, small_df):
        """Each row gets the churn rate of its category."""
        encoded = cls.encoder_helper(small_df, ["Gender", "Card_Category"], "Churn")
        assert encoded["Gender_Churn"].tolist() == pytest.approx([0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
        assert encoded["Card_Category_Churn"].tolist() == pytest.approx(
            [2 / 3, 0.0, 2 / 3, 2 / 3, 0.0, 0.0]
        )

    def test_uses_response_as_suffix(self, small_df):
        """New columns are named ``<column>_<response>``."""
        renamed = small_df.rename(columns={"Churn": "Target"})
        encoded = cls.encoder_helper(renamed, ["Gender"], "Target")
        assert "Gender_Target" in encoded.columns

    def test_does_not_modify_input(self, small_df):
        """The input dataframe is not changed in place."""
        before = small_df.copy()
        cls.encoder_helper(small_df, ["Gender"], "Churn")
        pd.testing.assert_frame_equal(small_df, before)

    def test_keeps_original_columns(self, small_df):
        """Original columns are preserved and exactly one column per category is added."""
        encoded = cls.encoder_helper(small_df, ["Gender", "Card_Category"], "Churn")
        assert list(encoded.columns[: small_df.shape[1]]) == list(small_df.columns)
        assert encoded.shape == (small_df.shape[0], small_df.shape[1] + 2)

    def test_empty_category_list(self, small_df):
        """An empty category list returns an equal dataframe."""
        pd.testing.assert_frame_equal(cls.encoder_helper(small_df, [], "Churn"), small_df)

    def test_all_project_categories(self, churn_df):
        """All project categories are encoded into values between 0 and 1 without NaNs."""
        encoded = cls.encoder_helper(churn_df, const.CAT_COLUMNS, const.RESPONSE)
        for column in const.CAT_COLUMNS:
            values = encoded[f"{column}_{const.RESPONSE}"]
            assert values.notna().all()
            assert values.between(0, 1).all()

    def test_unknown_column_raises(self, small_df):
        """Encoding a column that does not exist raises ``KeyError``."""
        with pytest.raises(KeyError):
            cls.encoder_helper(small_df, ["Not_A_Column"], "Churn")


class TestGetFeatureColumns:
    """Tests for ``get_feature_columns``."""

    def test_quantitative_then_encoded(self):
        """Quantitative columns come first, then one encoded column per category."""
        columns = cls.get_feature_columns("Churn")
        n_quant = len(const.QUANT_COLUMNS)
        assert columns[:n_quant] == const.QUANT_COLUMNS
        assert columns[n_quant:] == [f"{col}_Churn" for col in const.CAT_COLUMNS]

    def test_has_no_duplicates(self):
        """No feature appears twice."""
        columns = cls.get_feature_columns()
        assert len(columns) == len(set(columns))


class TestPerformFeatureEngineering:
    """Tests for ``perform_feature_engineering``."""

    def test_split_sizes(self, split_data, churn_df):
        """Train/test sizes follow ``TEST_SIZE`` and cover every row."""
        x_train, x_test, y_train, y_test = split_data
        assert len(x_train) + len(x_test) == len(churn_df)
        assert len(x_test) == pytest.approx(len(churn_df) * const.TEST_SIZE, abs=1)
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)

    def test_feature_columns(self, split_data):
        """X contains exactly the model features and never the response."""
        x_train, x_test, _, _ = split_data
        assert list(x_train.columns) == cls.get_feature_columns(const.RESPONSE)
        assert list(x_test.columns) == list(x_train.columns)
        assert const.RESPONSE not in x_train.columns

    def test_no_overlap_between_train_and_test(self, split_data):
        """No row ends up in both train and test sets."""
        x_train, x_test, _, _ = split_data
        assert set(x_train.index).isdisjoint(x_test.index)

    def test_is_reproducible(self, churn_df):
        """The split is deterministic thanks to the fixed random state."""
        first = cls.perform_feature_engineering(churn_df, const.RESPONSE)
        second = cls.perform_feature_engineering(churn_df, const.RESPONSE)
        assert first[0].index.equals(second[0].index)

    def test_encodes_when_needed(self, churn_df):
        """Unencoded input is encoded automatically and gives the same split."""
        encoded = cls.encoder_helper(churn_df, const.CAT_COLUMNS, const.RESPONSE)
        from_raw = cls.perform_feature_engineering(churn_df, const.RESPONSE)
        from_encoded = cls.perform_feature_engineering(encoded, const.RESPONSE)
        pd.testing.assert_frame_equal(from_raw[0], from_encoded[0])


class TestClassificationReportImage:
    """Tests for ``classification_report_image``."""

    def test_saves_both_reports(self, split_data):
        """One report image per model is written."""
        _, _, y_train, y_test = split_data
        cls.classification_report_image(y_train, y_test, y_train, y_train, y_test, y_test)
        _assert_image(const.RESULT_IMAGES["rf_report"])
        _assert_image(const.RESULT_IMAGES["lr_report"])

    def test_mismatched_lengths_raise(self, split_data):
        """Predictions with the wrong length raise ``ValueError``."""
        _, _, y_train, y_test = split_data
        with pytest.raises(ValueError):
            cls.classification_report_image(y_train, y_test, y_train[:-1], y_train, y_test, y_test)


class TestFeatureImportancePlot:
    """Tests for ``feature_importance_plot``."""

    def test_saves_plot(self, fitted_rf, split_data, tmp_path):
        """The plot is written to the requested path."""
        output = tmp_path / "importance.png"
        cls.feature_importance_plot(fitted_rf, split_data[1], str(output))
        _assert_image(output)

    def test_creates_missing_parent_directory(self, fitted_rf, split_data, tmp_path):
        """Missing parent folders of the output path are created."""
        output = tmp_path / "nested" / "dir" / "importance.png"
        cls.feature_importance_plot(fitted_rf, split_data[1], str(output))
        _assert_image(output)

    def test_model_without_importances_raises(self, split_data, tmp_path):
        """Models without ``feature_importances_`` raise ``AttributeError``."""
        x_train, _, y_train, _ = split_data
        model = LogisticRegression(max_iter=500).fit(x_train, y_train)
        with pytest.raises(AttributeError):
            cls.feature_importance_plot(model, x_train, str(tmp_path / "x.png"))


class TestShapSummaryPlot:
    """Tests for ``shap_summary_plot``."""

    def test_saves_plot(self, fitted_rf, split_data, tmp_path):
        """The SHAP summary is written to the requested path."""
        output = tmp_path / "shap.png"
        cls.shap_summary_plot(fitted_rf, split_data[1].head(30), str(output))
        _assert_image(output)


class TestRocCurvePlot:
    """Tests for ``roc_curve_plot``."""

    def test_saves_plot_with_named_curves(self, fitted_rf, split_data, tmp_path):
        """A labelled curve is drawn per model and the plot is saved."""
        x_train, x_test, y_train, y_test = split_data
        logistic = LogisticRegression(max_iter=500).fit(x_train, y_train)
        output = tmp_path / "roc.png"
        models = {"Random Forest": fitted_rf, "Logistic Regression": logistic}

        cls.roc_curve_plot(models, x_test, y_test, str(output))

        _assert_image(output)


class TestTrainModels:
    """Tests for ``train_models`` (with the shrunken hyper-parameter search)."""

    def test_saves_models_and_images(self, split_data):
        """Both models and every result image are written."""
        cls.train_models(*split_data)
        for path in const.MODEL_FILES.values():
            assert os.path.getsize(path) > 0
        for path in const.RESULT_IMAGES.values():
            _assert_image(path)

    def test_saved_models_predict(self, split_data):
        """The saved models load with joblib and predict binary labels for every test row."""
        _, x_test, _, _ = split_data
        cls.train_models(*split_data)
        for path in const.MODEL_FILES.values():
            predictions = joblib.load(path).predict(x_test)
            assert len(predictions) == len(x_test)
            assert set(predictions) <= {0, 1}

    def test_random_forest_beats_chance(self, split_data):
        """The tuned random forest learns the synthetic signal (accuracy well above chance)."""
        _, x_test, _, y_test = split_data
        cls.train_models(*split_data)
        model = joblib.load(const.MODEL_FILES["rfc"])
        assert model.score(x_test, y_test) > 0.8


class TestMain:
    """Tests for ``main`` (the full pipeline on synthetic data)."""

    def test_runs_end_to_end(self, csv_path, monkeypatch):
        """The pipeline produces every EDA image, result image and model."""
        monkeypatch.setattr(const, "DATA_PATH", csv_path)
        cls.main()
        artifacts = (
            list(const.EDA_IMAGES.values())
            + list(const.RESULT_IMAGES.values())
            + list(const.MODEL_FILES.values())
        )
        for path in artifacts:
            assert os.path.isfile(path), f"missing {path}"

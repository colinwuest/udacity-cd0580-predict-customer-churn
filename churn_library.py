"""Library of functions to identify credit card customers that are most likely to churn.

The module is a refactoring of ``churn_notebook.ipynb`` into reusable, documented
functions covering the full data-science process: data import, exploratory data
analysis (EDA), categorical encoding, feature engineering, model training,
prediction and evaluation. All figures and trained models are written to disk.

Leakage-free design: the data is split into train and test sets *before* any
encoding. The mean-response encoding of the categorical columns is a step inside
each model's scikit-learn ``Pipeline``, so it is learned from the training rows
only (and re-learned inside every cross-validation fold). The saved ``.pkl``
models therefore contain the encoding and predict directly from raw customer data.

Run the whole pipeline with::

    uv run python churn_library.py

Author: Colin Wuest
Date created: 2026-11-19
"""

import logging
import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

import constants as const

os.environ["QT_QPA_PLATFORM"] = "offscreen"
matplotlib.use("Agg")  # headless backend: figures are only saved, never shown
sns.set_theme()
logger = logging.getLogger(__name__)

# newer shap versions return one (samples, features, classes) array instead of a list
PER_CLASS_SHAP_NDIM = 3


def create_output_directories():
    """Create every output directory used by the project (idempotent).

    Args:
        None

    Returns:
        None
    """
    for directory in (const.EDA_DIR, const.RESULTS_DIR, const.MODELS_DIR):
        os.makedirs(directory, exist_ok=True)


def _save_figure(fig, output_pth):
    """Save a matplotlib figure to ``output_pth`` and release its memory.

    Args:
        fig: matplotlib Figure to save
        output_pth: destination file path

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_pth), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_pth, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure %s", output_pth)


def import_data(pth):
    """Return a dataframe for the csv found at ``pth`` with a binary churn column added.

    The binary response column (``Churn``: 1 = attrited, 0 = existing customer) is
    derived from ``Attrition_Flag`` so every downstream step can rely on it.

    Args:
        pth: path to the csv file

    Returns:
        df: pandas DataFrame
    """
    df = pd.read_csv(pth, index_col=0)
    df[const.RESPONSE] = (df[const.ATTRITION_COLUMN] != const.EXISTING_CUSTOMER).astype(int)
    logger.info("Imported %s with shape %s", pth, df.shape)
    return df


def perform_eda(df):
    """Perform EDA on ``df`` and save the figures to the EDA image folder.

    Saved plots:
        * univariate quantitative: churn histogram, customer age histogram,
          total transaction count density (with KDE)
        * univariate categorical: normalised marital status bar chart
        * bivariate: correlation heatmap of all numeric columns

    Args:
        df: pandas DataFrame containing the raw data and the response column

    Returns:
        None
    """
    create_output_directories()

    fig, ax = plt.subplots(figsize=(20, 10))
    df[const.RESPONSE].hist(ax=ax)
    ax.set(title="Churn distribution", xlabel="Churn (0 = existing, 1 = attrited)")
    _save_figure(fig, const.EDA_IMAGES["churn"])

    fig, ax = plt.subplots(figsize=(20, 10))
    df["Customer_Age"].hist(ax=ax)
    ax.set(title="Customer age distribution", xlabel="Customer age")
    _save_figure(fig, const.EDA_IMAGES["customer_age"])

    fig, ax = plt.subplots(figsize=(20, 10))
    df["Marital_Status"].value_counts(normalize=True).plot(kind="bar", ax=ax)
    ax.set(title="Marital status (share of customers)", ylabel="Share")
    _save_figure(fig, const.EDA_IMAGES["marital_status"])

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True, ax=ax)
    ax.set(title="Total transaction count distribution")
    _save_figure(fig, const.EDA_IMAGES["total_trans_ct"])

    fig, ax = plt.subplots(figsize=(20, 10))
    corr = df.select_dtypes(include="number").corr()
    sns.heatmap(corr, annot=False, cmap="Dark2_r", linewidths=2, ax=ax)
    ax.set(title="Correlation heatmap")
    _save_figure(fig, const.EDA_IMAGES["heatmap"])


def encoder_helper(category_lst, passthrough_lst=None):
    """Build the (unfitted) mean-response encoder for the categorical columns.

    Every column in ``category_lst`` is replaced by the churn rate of its category
    (scikit-learn ``TargetEncoder``); the numeric columns in ``passthrough_lst`` are
    kept unchanged and any other column (ids, the target, ...) is dropped. Adding a
    categorical feature only means extending the list.

    The encoder is meant to be the first step of a model ``Pipeline``. It must be
    fitted on training data only, so the churn rates never include test labels:

    * ``fit`` learns one churn rate per category from the training rows, with
      ``TARGET_ENCODER_SMOOTH`` shrinking rare categories towards the overall rate;
    * during ``fit_transform`` the training rows get out-of-fold (cross-fitted)
      encodings, so the model cannot memorise a row's own label;
    * ``transform`` maps new data with the learned rates; unseen categories get
      the overall training churn rate.

    Args:
        category_lst: list of categorical column names to encode
        passthrough_lst: list of numeric column names to keep as they are;
            defaults to ``constants.QUANT_COLUMNS``

    Returns:
        encoder: unfitted ``ColumnTransformer`` that outputs a pandas DataFrame with
            the encoded ``category_lst`` columns followed by ``passthrough_lst``
    """
    if passthrough_lst is None:
        passthrough_lst = const.QUANT_COLUMNS
    target_encoder = TargetEncoder(
        target_type="binary",
        smooth=const.TARGET_ENCODER_SMOOTH,
        cv=StratifiedKFold(
            n_splits=const.TARGET_ENCODER_CV_FOLDS,
            shuffle=True,
            random_state=const.RANDOM_STATE,
        ),
    )
    encoder = ColumnTransformer(
        [
            ("mean_response", target_encoder, list(category_lst)),
            ("numeric", "passthrough", list(passthrough_lst)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return encoder.set_output(transform="pandas")


def get_feature_columns():
    """Return the raw input columns the models are trained on and predict from.

    Args:
        None

    Returns:
        feature_columns: list of column names (``QUANT_COLUMNS`` + ``CAT_COLUMNS``)
    """
    return const.QUANT_COLUMNS + const.CAT_COLUMNS


def perform_feature_engineering(df, response):
    """Select the raw model features and split them into train and test sets.

    No encoding happens here: the categorical columns stay raw so that the
    encoding can be learned from the training split only (see ``encoder_helper``).
    The split is stratified on the response, so train and test sets have the
    same churn rate.

    Args:
        df: pandas DataFrame with the raw feature columns and the response column
        response: response column name

    Returns:
        x_train: DataFrame of raw training features
        x_test: DataFrame of raw test features
        y_train: Series of training responses
        y_test: Series of test responses
    """
    x_data = df[get_feature_columns()]
    y_data = df[response]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=const.TEST_SIZE,
        random_state=const.RANDOM_STATE,
        stratify=y_data,
    )
    logger.info(
        "Split data: %d train rows (churn rate %.3f), %d test rows (churn rate %.3f)",
        len(x_train),
        y_train.mean(),
        len(x_test),
        y_test.mean(),
    )
    return x_train, x_test, y_train, y_test


def build_models():
    """Build the two unfitted model pipelines, each starting with the encoder.

    Both pipelines encode the categorical columns themselves, so fitting them on
    training data keeps the test set out of the encoding, and the random forest
    search refits the encoder inside every cross-validation fold.

    Args:
        None

    Returns:
        cv_rfc: ``RandomizedSearchCV`` over an ``encode -> random forest`` pipeline
        lrc: ``encode -> StandardScaler -> LogisticRegression`` pipeline
    """
    rf_pipeline = Pipeline(
        [
            ("encode", encoder_helper(const.CAT_COLUMNS)),
            ("model", RandomForestClassifier(random_state=const.RANDOM_STATE)),
        ]
    ).set_output(transform="pandas")
    cv_rfc = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions={f"model__{key}": val for key, val in const.RF_PARAM_GRID.items()},
        n_iter=const.RF_SEARCH_ITERATIONS,
        cv=const.RF_SEARCH_CV_FOLDS,
        scoring="roc_auc",
        random_state=const.RANDOM_STATE,
        n_jobs=-1,
        error_score="raise",
    )
    lrc = Pipeline(
        [
            ("encode", encoder_helper(const.CAT_COLUMNS)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=const.LOGISTIC_MAX_ITER)),
        ]
    ).set_output(transform="pandas")
    return cv_rfc, lrc


def _unwrap_pipeline(model, x_data):
    """Split a fitted pipeline into its final estimator and its transformed input.

    Args:
        model: fitted ``Pipeline`` or a plain fitted estimator
        x_data: pandas DataFrame of raw input rows for ``model``

    Returns:
        estimator: the final estimator of the pipeline (or ``model`` itself)
        x_model: DataFrame as seen by ``estimator`` (encoded features for a pipeline,
            ``x_data`` unchanged otherwise)
    """
    if isinstance(model, Pipeline):
        return model[-1], model[:-1].transform(x_data)
    return model, x_data


def _report_image(model_name, train_report, test_report, output_pth):
    """Render train and test classification reports of one model into an image.

    Args:
        model_name: display name of the model
        train_report: classification report string for the training data
        test_report: classification report string for the test data
        output_pth: destination file path

    Returns:
        None
    """
    fig = plt.figure(figsize=(6, 6))
    text_kwargs = {"fontsize": 10, "fontproperties": "monospace"}
    fig.text(0.01, 0.95, f"{model_name} Train", **text_kwargs)
    fig.text(0.01, 0.55, train_report, **text_kwargs)
    fig.text(0.01, 0.45, f"{model_name} Test", **text_kwargs)
    fig.text(0.01, 0.05, test_report, **text_kwargs)
    plt.axis("off")
    os.makedirs(os.path.dirname(output_pth), exist_ok=True)
    fig.savefig(output_pth, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved classification report %s", output_pth)


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """Produce classification reports for train and test results and store them as images.

    Args:
        y_train: training response values
        y_test: test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest

    Returns:
        None
    """
    create_output_directories()
    reports = {
        "Random Forest": (y_train_preds_rf, y_test_preds_rf, const.RESULT_IMAGES["rf_report"]),
        "Logistic Regression": (
            y_train_preds_lr,
            y_test_preds_lr,
            const.RESULT_IMAGES["lr_report"],
        ),
    }
    for model_name, (train_preds, test_preds, output_pth) in reports.items():
        _report_image(
            model_name,
            classification_report(y_train, train_preds),
            classification_report(y_test, test_preds),
            output_pth,
        )


def feature_importance_plot(model, x_data, output_pth):
    """Create and store the feature importances of a tree-based model.

    Args:
        model: fitted model exposing ``feature_importances_``, or a fitted pipeline
            whose final step does
        x_data: pandas DataFrame of input rows for ``model`` (raw rows for a
            pipeline); used for the feature names
        output_pth: path to store the figure

    Returns:
        None
    """
    model, x_data = _unwrap_pipeline(model, x_data)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set(title="Feature Importance", ylabel="Importance")
    ax.bar(range(x_data.shape[1]), importances[indices])
    ax.set_xticks(range(x_data.shape[1]))
    ax.set_xticklabels(names, rotation=45, ha="right")
    _save_figure(fig, output_pth)


def shap_summary_plot(model, x_data, output_pth):
    """Create and store a SHAP bar summary plot for a tree-based model.

    Args:
        model: fitted tree-based model, or a fitted pipeline ending in one
        x_data: pandas DataFrame of input rows to explain (raw rows for a pipeline)
        output_pth: path to store the figure

    Returns:
        None
    """
    model, x_data = _unwrap_pipeline(model, x_data)
    shap_values = shap.TreeExplainer(model).shap_values(x_data)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == PER_CLASS_SHAP_NDIM:
        shap_values = [shap_values[:, :, idx] for idx in range(shap_values.shape[2])]

    plt.figure()
    shap.summary_plot(shap_values, x_data, plot_type="bar", show=False)
    # pylint wrongly infers that shap.summary_plot never returns
    # pylint: disable=unreachable
    ax = plt.gca()
    ax.set_xlabel("Mean |SHAP value| (feature importance)")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["Non-Churn (0)", "Churn (1)"], loc="lower right")
    _save_figure(plt.gcf(), output_pth)


def roc_curve_plot(models, x_test, y_test, output_pth):
    """Plot the ROC curves of several fitted models on one figure.

    Args:
        models: dict mapping a display name to a fitted classifier
        x_test: X testing data
        y_test: y testing data
        output_pth: path to store the figure

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(
            model, x_test, y_test, name=name, ax=ax, curve_kwargs={"alpha": 0.8}
        )
    ax.set(title="ROC curves")
    _save_figure(fig, output_pth)


def train_models(x_train, x_test, y_train, y_test):
    """Train, evaluate and store the models together with their result images.

    Both models are pipelines that start with the mean-response encoder (see
    ``build_models``). They are fitted on the training split only; the test split
    is used purely for evaluation. The random forest pipeline is tuned with a
    randomized, cross-validated search on the training data. Both fitted pipelines
    (encoder included) are saved with joblib, so they predict from raw customer
    rows. Classification reports, ROC curves, feature importances and a SHAP
    summary are saved as images.

    Args:
        x_train: DataFrame of raw training features
        x_test: DataFrame of raw test features
        y_train: Series of training responses
        y_test: Series of test responses

    Returns:
        None
    """
    create_output_directories()
    cv_rfc, lrc = build_models()

    logger.info("Training random forest pipeline (randomized search)...")
    cv_rfc.fit(x_train, y_train)
    logger.info(
        "Best random forest params: %s (CV ROC AUC on training data %.3f)",
        cv_rfc.best_params_,
        cv_rfc.best_score_,
    )
    logger.info("Training logistic regression pipeline...")
    lrc.fit(x_train, y_train)
    rfc = cv_rfc.best_estimator_

    for name, model in (("random forest", rfc), ("logistic regression", lrc)):
        auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
        logger.info("Test ROC AUC %s: %.3f", name, auc)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr=lrc.predict(x_train),
        y_train_preds_rf=rfc.predict(x_train),
        y_test_preds_lr=lrc.predict(x_test),
        y_test_preds_rf=rfc.predict(x_test),
    )
    roc_curve_plot(
        {"Logistic Regression": lrc, "Random Forest": rfc},
        x_test,
        y_test,
        const.RESULT_IMAGES["roc"],
    )
    feature_importance_plot(rfc, x_test, const.RESULT_IMAGES["feature_importance"])
    shap_summary_plot(rfc, x_test, const.RESULT_IMAGES["shap"])

    joblib.dump(rfc, const.MODEL_FILES["rfc"])
    joblib.dump(lrc, const.MODEL_FILES["logistic"])
    logger.info("Saved model pipelines (encoder included) to %s", const.MODELS_DIR)


def main():
    """Run the full churn pipeline end to end.

    Args:
        None

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    create_output_directories()
    df = import_data(const.DATA_PATH)
    perform_eda(df)
    x_train, x_test, y_train, y_test = perform_feature_engineering(df, const.RESPONSE)
    train_models(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()

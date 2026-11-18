"""Library of functions to identify credit card customers that are most likely to churn.

The module is a refactoring of ``churn_notebook.ipynb`` into reusable, documented
functions covering the full data-science process: data import, exploratory data
analysis (EDA), categorical encoding, feature engineering, model training,
prediction and evaluation. All figures and trained models are written to disk.

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def encoder_helper(df, category_lst, response):
    """Mean-response (target) encode each categorical column in ``category_lst``.

    For every column ``col`` a new column ``f"{col}_{response}"`` is added that
    holds the churn rate of the category the row belongs to. This replaces the
    copy-pasted blocks of the notebook with a single loop, so adding a feature
    only means extending the list.

    Args:
        df: pandas DataFrame
        category_lst: list of categorical column names to encode
        response: name of the response column (also used as the new-column suffix)

    Returns:
        df: copy of the DataFrame with the encoded columns added
    """
    df = df.copy()
    for column in category_lst:
        df[f"{column}_{response}"] = df[column].map(df.groupby(column)[response].mean())
    logger.info("Encoded categorical columns: %s", ", ".join(category_lst))
    return df


def get_feature_columns(response=const.RESPONSE):
    """Return the ordered list of model input columns.

    Args:
        response: response column name used as suffix for encoded categorical columns

    Returns:
        list of column names (quantitative + encoded categorical)
    """
    return const.QUANT_COLUMNS + [f"{col}_{response}" for col in const.CAT_COLUMNS]


def perform_feature_engineering(df, response):
    """Select the model features and split the data into train and test sets.

    Categorical columns are encoded here if ``encoder_helper`` has not been run yet.

    Args:
        df: pandas DataFrame
        response: response column name

    Returns:
        x_train: X training data
        x_test: X testing data
        y_train: y training data
        y_test: y testing data
    """
    feature_cols = get_feature_columns(response)
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        df = encoder_helper(df, const.CAT_COLUMNS, response)

    x_data = df[feature_cols]
    y_data = df[response]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=const.TEST_SIZE, random_state=const.RANDOM_STATE
    )
    logger.info("Split data: %d train rows, %d test rows", len(x_train), len(x_test))
    return x_train, x_test, y_train, y_test


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
        model: fitted model exposing ``feature_importances_``
        x_data: pandas DataFrame of X values (used for the feature names)
        output_pth: path to store the figure

    Returns:
        None
    """
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
        model: fitted tree-based model
        x_data: pandas DataFrame of X values to explain
        output_pth: path to store the figure

    Returns:
        None
    """
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

    A random forest is tuned with a randomized hyper-parameter search, and a
    standardised logistic regression is fitted as baseline. Both models are
    saved with joblib; classification reports, ROC curves, feature importances
    and a SHAP summary are saved as images.

    Args:
        x_train: X training data
        x_test: X testing data
        y_train: y training data
        y_test: y testing data

    Returns:
        None
    """
    create_output_directories()

    cv_rfc = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=const.RANDOM_STATE),
        param_distributions=const.RF_PARAM_GRID,
        n_iter=const.RF_SEARCH_ITERATIONS,
        cv=const.RF_SEARCH_CV_FOLDS,
        random_state=const.RANDOM_STATE,
        n_jobs=-1,
        error_score="raise",
    )
    lrc = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=const.LOGISTIC_MAX_ITER)),
        ]
    )

    logger.info("Training random forest (randomized search)...")
    cv_rfc.fit(x_train, y_train)
    logger.info("Best random forest params: %s", cv_rfc.best_params_)
    logger.info("Training logistic regression...")
    lrc.fit(x_train, y_train)
    rfc = cv_rfc.best_estimator_

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
    logger.info("Saved models to %s", const.MODELS_DIR)


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
    df = encoder_helper(df, const.CAT_COLUMNS, const.RESPONSE)
    x_train, x_test, y_train, y_test = perform_feature_engineering(df, const.RESPONSE)
    train_models(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()

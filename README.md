# Predict Customer Churn

[![CI](https://github.com/colinwuest/udacity-cd0580-predict-customer-churn/actions/workflows/ci.yml/badge.svg)](https://github.com/colinwuest/udacity-cd0580-predict-customer-churn/actions/workflows/ci.yml)
[![Train models](https://github.com/colinwuest/udacity-cd0580-predict-customer-churn/actions/workflows/train.yml/badge.svg)](https://github.com/colinwuest/udacity-cd0580-predict-customer-churn/actions/workflows/train.yml)

- Project **Predict Customer Churn** of the ML DevOps Engineer Nanodegree (Udacity)
- Repository platform: **GitHub** (https://github.com/colinwuest/udacity-cd0580-predict-customer-churn)

---

## Project Description

Banks lose revenue when credit card customers leave (churn). This project
predicts which customers are likely to churn, so the bank can contact them
before they leave.

The original solution was an exploratory Jupyter notebook (`churn_notebook.ipynb`).
This project turns it into production-style code: a modular, documented, PEP 8
compliant library with tests and logging, and with every figure and model saved to disk.

The machine learning pipeline:

1. **Import** the bank customer data and create the binary target `Churn`
   (1 = `Attrited Customer`, 0 = `Existing Customer`).
2. **Exploratory data analysis (EDA)**: saves univariate quantitative, univariate
   categorical and bivariate (correlation heatmap) plots.
3. **Feature engineering**: mean-response (target) encoding of the five categorical
   columns in one loop, then a 70/30 train/test split.
4. **Model training**:
   - **Random Forest**, tuned with `RandomizedSearchCV` (12 candidates, 3-fold CV)
   - **Logistic Regression** in a `StandardScaler` pipeline (baseline)
5. **Prediction and evaluation**: classification reports, ROC curves, Random Forest
   feature importances and a SHAP summary plot, all saved as images.
6. **Persistence**: both models are saved as `.pkl` files with `joblib`.

On the test set the Random Forest reaches about 0.99 ROC AUC and the Logistic
Regression about 0.92.

---

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for Python and dependency
management and [ruff](https://docs.astral.sh/ruff/) for formatting and linting.
Requires Python 3.12 or 3.13 (uv downloads it automatically if needed).

```bash
# install uv once (any of these)
pip install uv            # or: pipx install uv / winget install astral-sh.uv / brew install uv

# create .venv and install all runtime + dev dependencies from uv.lock
uv sync
```

Without uv, use plain pip with the exported lock file:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Main libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, shap, joblib.
Dev tools: ruff, pylint, pytest.

### Dev container (Ubuntu, Docker)

The `.devcontainer/` folder defines a reproducible Ubuntu 24.04 environment with
Python 3.12, uv and all runtime and dev dependencies preinstalled from `uv.lock`.

- **VS Code**: install the *Dev Containers* extension, open the folder, and run
  **Dev Containers: Reopen in Container**. Ruff, Pylint, Python and Jupyter extensions
  are configured automatically.
- **GitHub Codespaces**: *Code → Codespaces → Create codespace on main*.
- **Plain Docker** (no VS Code):

  ```bash
  docker build -f .devcontainer/Dockerfile -t churn-dev .
  docker run --rm -it -v "$PWD":/workspaces/churn -w /workspaces/churn churn-dev \
      uv run python churn_script_logging_and_tests.py
  ```

The virtual environment lives at `/opt/venv` inside the container
(`UV_PROJECT_ENVIRONMENT`), so it never conflicts with a `.venv` on the host.
All `uv run ...` commands below work unchanged inside the container.

> **Windows note:** shap and matplotlib have no wheels for 32-bit Python. If your
> system Python is 32-bit, let uv use a 64-bit interpreter:
> `uv python pin cpython-3.12-windows-x86_64-none` and then `uv sync`.

---

## Running the Files

### 1. Run the Pipeline

```bash
uv run python churn_library.py
```

What happens:

1. `data/bank_data.csv` is loaded and the `Churn` target is created.
2. EDA plots are saved to `images/eda/`.
3. Categorical columns are mean-encoded, then the data is split 70/30.
4. A Random Forest (randomized hyper-parameter search) and a Logistic Regression are trained.
5. Classification reports, ROC curves, feature importances and the SHAP summary are
   saved to `images/results/`.
6. Both models are saved to `models/`.

Progress is printed to the console. The run takes about 30–60 seconds on a laptop.

---

### 2. Run Tests and Logging

```bash
uv run python churn_script_logging_and_tests.py
```

This runs the five tests in pipeline order (`test_import`, `test_eda`,
`test_encoder_helper`, `test_perform_feature_engineering`, `test_train_models`) and
writes a fresh `logs/churn_library.log`. For each test the log contains:

- `Running test_<name>`: which test started
- `SUCCESS test_<name>: ...` (INFO): what was verified (shapes, number of files saved)
- `ERROR test_<name>: ...` (ERROR): what failed and why. The exception is re-raised,
  so the script stops with a non-zero exit code.

Detailed INFO messages from `churn_library` (files saved, best hyper-parameters,
etc.) are written to the same log file.

The same tests also run under pytest:

```bash
uv run pytest
```

### 3. Unit Tests

`tests/test_churn_library.py` contains 36 fast unit tests (about 20 s in total)
that test every function in `churn_library.py` on its own:

- **Synthetic data**: a 240-row dataset with the same schema as `bank_data.csv`,
  so the real data is never needed.
- **Isolated outputs**: every image and model path is redirected to a pytest
  temporary folder, so the real `images/` and `models/` folders are never modified.
- **Fast training**: the random forest search is shrunk to one small candidate.
- **What is covered**: correct values (e.g. hand-computed churn rates for
  `encoder_helper`, the `Churn` mapping), edge cases (missing file, missing or unknown
  columns, empty category list, mismatched prediction lengths, models without
  `feature_importances_`), side effects (input not mutated, figures closed, parent
  folders created) and saved models that load and predict.

```bash
uv run pytest tests                     # unit tests only (fast)
uv run pytest -m "not integration"      # same, by marker
uv run pytest                           # unit + end-to-end tests on the real data
```

The tests in `churn_script_logging_and_tests.py` are marked `integration`
automatically by the root `conftest.py`.

`tests/test_docstrings.py` checks the rubric's documentation requirements without
running any project code. It parses every learner-written file and fails if:

- a file docstring lacks a purpose, `Author:` or `Date created:`;
- a function (including fixtures and tests) has no purpose line;
- a function has no `Args:` section naming each input, or `None` when it has no inputs;
- a function has no `Returns:` / `Yields:` section, or `None` when it returns nothing.

---

### 4. Code Quality

```bash
uv run ruff format .                    # auto-format (PEP 8, line length 100)
uv run ruff check .                     # lint: pycodestyle, pyflakes, isort, naming, docstrings, ...
uv run pylint churn_library.py churn_script_logging_and_tests.py tests
```

Both files score **10.00/10** with pylint and pass `ruff check`. pylint loads the
`docparams` extension, so an undocumented parameter, return value or yield in any
function fails the lint. The configuration lives in `pyproject.toml`.

---

### 5. CI/CD (GitHub Actions)

Three workflows in `.github/workflows/` run on GitHub-hosted Ubuntu runners with uv:

| Workflow | Trigger | What it does |
|---|---|---|
| **CI** (`ci.yml`) | every push to `main`, every pull request, manual | `lint`: `ruff format --check`, `ruff check`, pylint (fails below 8.5) → `unit-tests`: `pytest tests` on Python 3.12 and 3.13 → `integration-tests`: runs `churn_script_logging_and_tests.py` on the real data and uploads `logs/churn_library.log` as the **test-log** artifact |
| **Train models** (`train.yml`) | manual (*Actions → Train models → Run workflow*), and pushes to `main` that change the pipeline, constants, data or dependencies | runs `churn_library.py`, checks the models and plots exist, writes the best hyper-parameters to the job summary, and uploads the **models** (`*.pkl`) and **images-and-log** artifacts (kept 30 days) |
| **Release** (`release.yml`) | push of a `v*` tag, or manual | lint + unit tests + test script, then builds and publishes the release zip with the repo and all rubric outputs (see below) |

Because generated files are not committed, download the trained models and plots
from the artifacts of the latest *Train models* run, or from a release.

#### Releases (`release.yml`)

The **Release** workflow builds `predict-customer-churn-<version>.zip`. The zip contains
every tracked file of the repository plus all outputs the rubric asks for, all
generated fresh on the runner:

| Folder in the zip | Contents |
|---|---|
| `images/eda/` | EDA plots (univariate quantitative, univariate categorical, bivariate) |
| `images/results/` | classification reports, ROC curves, feature importance, SHAP summary |
| `models/` | `rfc_model.pkl`, `logistic_model.pkl` |
| `logs/` | `churn_library.log` written by `churn_script_logging_and_tests.py` |
| `reports/` | `ruff.txt`, `pylint.txt` (score), `unit-tests.txt` / `.xml` |
| `RELEASE_INFO.md` | version, commit and build details |

The workflow fails if any rubric output is missing or the test log contains an
`ERROR`. The zip and its `.sha256` checksum are attached to the GitHub release.

```bash
git tag v1.0.0
git push origin v1.0.0       # builds and publishes release v1.0.0
```

Alternatively, run it manually under *Actions → Release → Run workflow* and enter a
version. Leave **publish** unticked for a dry run that only uploads the zip as a
build artifact.

---
## Files and Data Description

```
.
├── .github/workflows/                # CI (lint + tests), model training and release pipelines
├── churn_library.py                  # the refactored ML pipeline (library + entry point)
├── churn_script_logging_and_tests.py # tests + logging for every library function
├── conftest.py                       # pytest fixtures (lets the tests run under pytest too)
├── constants.py                      # paths, column lists, hyper-parameters, artifact names
├── tests/                            # fast pytest unit tests on synthetic data
│   ├── conftest.py                   #   synthetic dataset + temporary output folders
│   └── test_churn_library.py         #   unit tests per library function
├── churn_notebook.ipynb              # original exploratory notebook (reference solution)
├── guide.ipynb                       # project instructions
├── pyproject.toml / uv.lock          # project metadata + locked dependencies (uv)
├── requirements.txt                  # pinned dependencies exported from uv.lock (pip fallback)
├── data/bank_data.csv                # input data
├── images/eda/                       # EDA plots (generated, git-ignored)
├── images/results/                   # model evaluation plots (generated, git-ignored)
├── models/                           # trained models (generated, git-ignored)
└── logs/churn_library.log            # test/log output (generated, git-ignored)
```

### Main Files

- `churn_library.py`
  The ML pipeline as reusable functions: `import_data`, `perform_eda`,
  `encoder_helper`, `perform_feature_engineering`, `classification_report_image`,
  `feature_importance_plot`, `train_models` and a few plotting helpers
  (`roc_curve_plot`, `shap_summary_plot`). Running it as a script runs the whole
  pipeline end to end.

- `churn_script_logging_and_tests.py`
  One test per library function. Each test checks that returned objects are not
  empty (shapes, columns, value ranges) and that expected artifacts (images, models)
  are created. Every test logs `SUCCESS` (INFO) or `ERROR` messages to
  `logs/churn_library.log`.

- `constants.py`
  All configuration in one place. To add a new categorical feature, append its
  name to `CAT_COLUMNS`. Encoding, feature selection and the tests pick it up
  automatically.

- `churn_notebook.ipynb`
  The original notebook that this project refactors. It is kept as a reference and
  is not used by the scripts.

---

### Data

- `data/bank_data.csv`: 10,127 credit card customers, 22 columns
  ([Kaggle: Credit Card Customers](https://www.kaggle.com/sakshigoyal7/credit-card-customers)).
  - **Target**: `Attrition_Flag` (`Existing Customer` / `Attrited Customer`), turned into
    the binary column `Churn`. About 16 % of customers churned.
  - **Categorical features (5)**: `Gender`, `Education_Level`, `Marital_Status`,
    `Income_Category`, `Card_Category`
  - **Quantitative features (14)**: e.g. `Customer_Age`, `Credit_Limit`,
    `Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Revolving_Bal`, `Avg_Utilization_Ratio`
  - `CLIENTNUM` is an ID and is not used as a feature. The dataset has no missing values.

---

### Output Directories

After running the project, outputs are saved to:

- EDA images → `images/eda/`
- Model results → `images/results/`
- Models → `models/`
- Logs → `logs/churn_library.log`

These folders are **not committed** (see `.gitignore`), so a fresh clone contains
none of them. Run the pipeline or the test script once to create them. The code
creates every folder it needs.

---
## Expected Outputs

- Models
  - `models/rfc_model.pkl`: best Random Forest from the randomized search
  - `models/logistic_model.pkl`: scaler + Logistic Regression pipeline

  Load them with `joblib.load(path)` and call `.predict(X)`. `X` must contain the
  columns returned by `churn_library.get_feature_columns()`.

- EDA images (`images/eda/`)
  - `churn_distribution.png`: histogram of the target (univariate, quantitative)
  - `customer_age_distribution.png`: histogram of customer age (univariate, quantitative)
  - `marital_status_distribution.png`: normalised bar chart (univariate, categorical)
  - `total_transaction_distribution.png`: density + KDE of `Total_Trans_Ct`
  - `heatmap.png`: correlation heatmap of numeric columns (bivariate)

- Model result images (`images/results/`)
  - `rf_results.png`, `logistic_results.png`: train/test classification reports
  - `roc_curve_result.png`: ROC curves of both models
  - `feature_importances.png`: Random Forest feature importances
  - `shap_summary.png`: mean |SHAP value| per feature and class

- Logs
  - `logs/churn_library.log`

---

## Notes

- **Logging**: the test script attaches a file handler to the root logger, so it
  captures both the test messages (`churn_tests` logger) and the library messages
  (`churn_library` logger). It also works under pytest. Running the script directly
  overwrites the log; running pytest appends to it.
- **Target encoding**: categories are encoded with the mean churn rate computed on
  the full dataset, as in the original notebook. This leaks a little target information
  into the test set. A production version should fit the encoding on the training
  split only, e.g. with `sklearn.preprocessing.TargetEncoder`.
- **Class imbalance**: only ~16 % of customers churn. Recall on the churn class
  (~0.76 for the Random Forest on the test set) matters more than accuracy.
  Class weights or threshold tuning could improve it.
- **Headless plotting**: matplotlib uses the `Agg` backend, so the code runs on
  servers or CI machines without a display.
- `test_train_models` retrains the models (about 30 s) and overwrites the files in `models/`.

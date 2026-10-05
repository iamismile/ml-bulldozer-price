# 🚜 Bulldozer Sale Price Prediction

A machine learning project that predicts the **sale price of a bulldozer** from its characteristics (year made, model, state of sale, drive system, and more) and its sale date.

Built with a **Random Forest Regressor** on the [Blue Book for Bulldozers](https://www.kaggle.com/competitions/bluebook-for-bulldozers) Kaggle dataset.

|                  |                                                              |
| ---------------- | ------------------------------------------------------------ |
| **Inputs**       | Bulldozer characteristics + sale date                        |
| **Output**       | Sale price in USD                                            |
| **Problem type** | Regression with a time component (past sales → future sales) |
| **Metric**       | RMSLE (Root Mean Squared Log Error), lower is better         |
| **Model**        | `RandomForestRegressor` (Scikit-Learn)                       |

---

## 📚 Table of Contents

- [Problem Definition](#-problem-definition)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Workflow](#-workflow)
- [Key Design Decisions](#-key-design-decisions)
- [Outputs](#-outputs)
- [Results](#-results)
- [Ideas for Improvement](#-ideas-for-improvement)
- [Acknowledgements](#-acknowledgements)

---

## 🎯 Problem Definition

> **How well can we predict the future sale price of a bulldozer, given its characteristics and previous sales?**

Because we predict a number, this is a **regression** problem. Because we use past sales to predict future ones, it is also a **time series forecasting** problem, so the train/validation split must respect time order.

---

## 📦 Dataset

The data comes from the Blue Book for Bulldozers competition on Kaggle.

- [Dataset on Kaggle](https://www.kaggle.com/competitions/bluebook-for-bulldozers/data)
- [Direct download (course GitHub)](https://github.com/mrdbourke/zero-to-mastery-ml/raw/refs/heads/master/data/bluebook-for-bulldozers.zip)
- [Data dictionary (Google Sheet)](https://docs.google.com/spreadsheets/d/18ly-bLR8sbDJLITkWG7ozKm8l3RyieQ2Fpgix-beSYI/edit?gid=1021421956#gid=1021421956)

| File                | Contents                                         | Used for          |
| ------------------- | ------------------------------------------------ | ----------------- |
| `Train.csv`         | Sales up to the end of 2011                      | Training          |
| `Valid.csv`         | Sales from Jan 1 to Apr 30, 2012 (features only) | Validation        |
| `ValidSolution.csv` | True sale prices for `Valid.csv`                 | Validation labels |
| `Test.csv`          | Sales from May 1 to Nov 2012 (no prices)         | Final predictions |
| `TrainAndValid.csv` | Train + Valid combined                           | Exploration only  |

> The dataset is subject to Kaggle's competition terms and is **not** included in this repository. Download it and place the files as shown in [Project Structure](#-project-structure).

---

## 🗂 Project Structure

```text
.
├── bulldozer_price_prediction.ipynb     # Main notebook (all steps, with explanations)
├── main.py                              # Default uv entry-point stub (not used by the notebook)
├── pyproject.toml                       # Project metadata + dependencies (managed by uv)
├── uv.lock                              # Locked versions for reproducibility
├── README.md
├── models/
│   └── randomforest_regressor_best_RMSLE.pkl      # Saved best model (created by the notebook)
└── data/
    └── bluebook-for-bulldozers/
        ├── Train.csv
        ├── Valid.csv
        ├── ValidSolution.csv
        ├── Test.csv
        ├── TrainAndValid.csv
        ├── Data Dictionary.xlsx                    # Column descriptions
        ├── TrainAndValid_object_values_as_categories.parquet   # Checkpoint (created by the notebook)
        └── bulldozer_predictions.csv               # Test predictions (created by the notebook)
```

The CSV/XLSX files come with the Kaggle download. The notebook creates the `.parquet` checkpoint, the saved model in `models/`, and the predictions file.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Install dependencies with [uv](https://docs.astral.sh/uv/)

This project uses **uv** for environment and dependency management and requires **Python 3.11 or newer**. If you don't have uv yet, see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

This creates the `.venv` virtual environment and installs the exact versions pinned in `uv.lock`.

| Package                | Minimum version | Why it's needed                                  |
| ---------------------- | --------------- | ------------------------------------------------ |
| `numpy`, `pandas`      | 2.4.6, 3.0.6    | Data handling                                    |
| `matplotlib`           | 3.11.2          | Plots                                            |
| `seaborn`              | 0.13.2          | Optional plotting (not used by the notebook yet) |
| `scikit-learn`         | 1.9.1           | Preprocessing, model, tuning, metrics            |
| `joblib`               | 1.6.0           | Saving / loading the trained model               |
| `pyarrow`              | 25.0.1          | Reading / writing Parquet checkpoints            |
| `jupyter`, `ipykernel` | 1.1.1, 7.3.0    | Running the notebook                             |

To add another package later: `uv add <package-name>`.

### 3. Download the data

Download the dataset (links above), unzip it, and place the CSV files in `data/bluebook-for-bulldozers/`. If you keep them elsewhere, change the `DATA_DIR` variable at the top of the notebook.

### 4. Run the notebook

```bash
uv run jupyter notebook bulldozer_price_prediction.ipynb
```

`uv run` launches Jupyter inside the project's environment, so no manual activation is needed. (In VS Code, just select the `.venv` interpreter as the notebook kernel.) Then run the cells from top to bottom.

> ⏱ **Heads-up:** the `RandomizedSearchCV` step (20 iterations × 3-fold CV) and the final 500-tree model can take a while on a laptop. Lower `n_iter`, `n_estimators`, or keep `max_samples=10000` for faster experiments.

---

## 🔄 Workflow

1. **Frame the problem:** define inputs, outputs and the evaluation metric (RMSLE).
2. **Explore the data:** missing values, price distribution, sales over time.
3. **Parse and sort dates:** convert `saledate` to datetime and order chronologically.
4. **Feature engineering:** extract `saleYear`, `saleMonth`, `saleDay`, `saleDayofweek`, `saleDayofyear`.
5. **Prototype with Pandas:** categories to codes, median fill, `_is_missing` flags (and why this leaks data).
6. **Time-based split:** use `Train.csv` and `Valid.csv` instead of a random split.
7. **Preprocess with Scikit-Learn:** `OrdinalEncoder` + training medians, learned from training data only.
8. **Baseline model:** `RandomForestRegressor` with `max_samples=10000`.
9. **Evaluate:** MAE, RMSLE and R² on training and validation sets.
10. **Tune hyperparameters:** `RandomizedSearchCV`.
11. **Compare models:** table and bar chart of validation RMSLE.
12. **Save and load:** persist the best model with `joblib`.
13. **Feature importance:** see which columns drive the predictions.
14. **Predict on the test set:** apply identical preprocessing and write `bulldozer_predictions.csv`.

---

## 🧠 Key Design Decisions

- **Time-based split, not random.** The validation set (Jan-Apr 2012) represents the future, so the model never trains on data from after the samples it is evaluated on.
- **No data leakage.** Category mappings and medians are learned from the **training set only** and then applied unchanged to validation and test data.
- **Missing-value flags.** Numeric columns with missing values get an extra `<column>_is_missing` column, so the model keeps the information that a value was absent.
- **Unknown categories → `-1`.** `OrdinalEncoder` encodes missing and previously unseen categories as `-1`, so new data never crashes the pipeline.
- **One reusable `preprocess()` function** is applied to train, validation and test data, keeping all three consistent.
- **Parquet checkpoints.** Parquet preserves Pandas dtypes (such as `category`), which CSV does not.
- **RMSLE as the metric.** It measures _relative_ error on log prices, which suits a right-skewed price distribution.

---

## 📤 Outputs

| File                                                     | Description                                                    |
| -------------------------------------------------------- | -------------------------------------------------------------- |
| `models/randomforest_regressor_best_RMSLE.pkl`           | Trained model with the best hyperparameters                    |
| `data/bluebook-for-bulldozers/bulldozer_predictions.csv` | Predictions for `Test.csv` with columns `SalesID`, `SalePrice` |

> To reuse the model on new data later, also save the fitted `ordinal_encoder` and `train_medians`, since they are part of the preprocessing.

---

## 📊 Results

Scores on the validation set (Jan-Apr 2012), the "future" data the model never trained on. Lower is better for RMSLE and MAE; higher is better for R².

| Model                   | Validation RMSLE ↓ | Validation MAE ↓ | Validation R² ↑ |
| ----------------------- | -----------------: | ---------------: | --------------: |
| Default Random Forest   |             0.2938 |           $7,133 |           0.835 |
| Randomized search model |             0.2967 |           $7,215 |           0.832 |
| **Ideal (tuned) model** |         **0.2425** |       **$5,868** |       **0.884** |

Full training vs. validation comparison:

| Model                   | Training RMSLE | Validation RMSLE | Training MAE | Validation MAE | Training R² | Validation R² |
| ----------------------- | -------------: | ---------------: | -----------: | -------------: | ----------: | ------------: |
| Default Random Forest   |         0.2577 |           0.2938 |       $5,558 |         $7,133 |       0.861 |         0.835 |
| Randomized search model |         0.2674 |           0.2967 |       $5,834 |         $7,215 |       0.848 |         0.832 |
| **Ideal (tuned) model** |         0.1270 |       **0.2425** |       $2,496 |         $5,868 |       0.969 |         0.884 |

**Takeaways**

- The **ideal model** is the best on every validation metric, with an RMSLE of about **0.24** and an average error of about **$5,900** per bulldozer.
- The randomized-search model scored about the same as the default. Both trained each tree on only 10,000 rows (`max_samples=10000`), while the ideal model uses all training rows per tree, so part of its improvement likely comes from that and not from the other hyperparameters alone.
- The ideal model fits the training data much more closely than the validation data (R² 0.969 vs 0.884). Some overfitting is normal for Random Forests, but it shows there is room to improve generalisation.

Exact numbers vary with hardware, library versions and random seeds.

---

## 💡 Ideas for Improvement

- Train on `log(SalePrice)` directly, since RMSLE is a log-based metric.
- Use `TimeSeriesSplit` for cross-validation so folds respect time order.
- Try other models such as `HistGradientBoostingRegressor`, XGBoost, LightGBM or CatBoost.
- Drop low-importance features and compare validation RMSLE.
- Wrap preprocessing and model in a Scikit-Learn `Pipeline` with a `ColumnTransformer`.
- Add more date-based features, such as machine age at the time of sale.

---

## 🙏 Acknowledgements

- [Blue Book for Bulldozers](https://www.kaggle.com/competitions/bluebook-for-bulldozers) on Kaggle for the dataset.
- The [Zero to Mastery ML course](https://github.com/mrdbourke/zero-to-mastery-ml) for the original project structure this notebook is based on.

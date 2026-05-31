# Banking Customer Churn Prediction — XGBoost + Tidymodels (R)

An end-to-end machine learning pipeline in R that predicts bank customer churn using **GPU-accelerated XGBoost**, a fully declarative **tidymodels** preprocessing recipe, **SMOTE** class-imbalance correction, and a **two-stage Bayesian hyperparameter search** — all rendered as a reproducible HTML report via R Markdown.

> Built with: `tidymodels` · `xgboost (CUDA)` · `themis` · `bonsai` · `healthyR.ai` · R

---

## Results

| Metric | Score |
|---|---|
| Accuracy | **0.969** |
| Sensitivity (Recall) | **0.878** |
| Specificity | **0.986** |
| Precision (PPV) | **0.921** |
| F1 Score | **0.899** |
| ROC AUC | optimised (selection criterion) |
| Balanced Accuracy | reported |
| MCC | reported |

Sensitivity of **0.878** on a dataset with **83.9% majority class** — without SMOTE correction the naive model would simply predict "Existing Customer" always and achieve 83.9% accuracy with zero recall on churners. These metrics confirm the model is genuinely learning churn signal rather than exploiting class imbalance.

---

## Dataset

| Property | Detail |
|---|---|
| Source | `BankChurners.csv` |
| Observations | 10,127 customers |
| Raw columns | 22 (21 features + target) |
| Target variable | `Attrition_Flag` — `"Existing Customer"` / `"Attrited Customer"` |
| Class distribution | 83.9% Existing · 16.1% Attrited (~5:1 imbalance) |
| Numeric features | 15 columns |
| Categorical features | 6 columns |

**All 21 feature columns:**

| Column | Type | Description |
|---|---|---|
| `CLIENTNUM` | numeric | Customer identifier (excluded from modelling) |
| `Customer_Age` | numeric | Age of the customer |
| `Gender` | character | M / F |
| `Dependent_count` | numeric | Number of dependents |
| `Education_Level` | character | 7 levels — collapsed during preprocessing |
| `Marital_Status` | character | 4 levels — collapsed during preprocessing |
| `Income_Category` | character | 6 income bands — collapsed during preprocessing |
| `Card_Category` | character | Card tier (Blue / Silver / Gold / Platinum) |
| `Months_on_book` | numeric | Customer tenure in months |
| `Total_Relationship_Count` | numeric | Number of products held |
| `Months_Inactive_12_mon` | numeric | Months inactive in last 12 months |
| `Contacts_Count_12_mon` | numeric | Contact events in last 12 months |
| `Credit_Limit` | numeric | Credit card limit |
| `Total_Revolving_Bal` | numeric | Total revolving balance |
| `Avg_Open_To_Buy` | numeric | Average open-to-buy credit line remaining |
| `Total_Amt_Chng_Q4_Q1` | numeric | Change in transaction amount (Q4 vs Q1) |
| `Total_Trans_Amt` | numeric | Total transaction amount (last 12 months) |
| `Total_Trans_Ct` | numeric | Total transaction count (last 12 months) |
| `Total_Ct_Chng_Q4_Q1` | numeric | Change in transaction count (Q4 vs Q1) |
| `Avg_Utilization_Ratio` | numeric | Average card utilisation ratio |

---

## Pipeline Architecture

```
BankChurners.csv (10,127 rows × 22 cols)
        │
        ▼
initial_split(prop = 0.8, seed = 123)
        ├── df_train (8,101 rows)
        └── df_test  (2,026 rows)
        │
        ▼
recipe(Attrition_Flag ~ ., data = df_train)
        │
        ├── Step 1: NA Analysis
        │       colSums(is.na(df)) → bar plot
        │       update_role(col_remove, new_role = "columns_to_remove")
        │       → removes high-NA columns from modelling
        │
        ├── Step 2: Outlier Detection + Winsorization
        │       IQR method: lower = Q1 − 1.5×IQR, upper = Q3 + 1.5×IQR
        │       count_outlier() → outlier.df (per column)
        │       step_hai_winsorized_truncate(outlier_cols, fraction = 0.05)
        │       → clips extreme values at 5th/95th percentile
        │
        ├── Step 3: Scale Diagnosis
        │       log10(|Mean − SD| + 1e-12) per numeric column → bar plot
        │       step_normalize(all_numeric_predictors())
        │       → zero-mean, unit-variance normalisation
        │
        ├── Step 4: Categorical Semantic Collapsing
        │       Income_Category:  6 levels → 3
        │           "Less than $40K"              → "Less than 40K"
        │           "$120K +"                     → "More than 120K"
        │           "$40K-$60K","$60K-$80K",
        │            "$80K-$120K","Unknown"        → "Average Between 40K to 120K"
        │
        │       Marital_Status:   4 levels → 2
        │           "Married"                     → Married
        │           "Single","Divorced","Unknown"  → Single
        │
        │       Education_Level:  7 levels → 2
        │           "Uneducated","Unknown"         → Uneducated
        │           "High School","Graduate",
        │           "College","Post-Graduate",
        │           "Doctorate"                   → Educated
        │
        │       step_novel() applied before each collapse
        │       → handles unseen categories at inference time
        │
        ├── Step 5: One-Hot Encoding
        │       step_dummy(c("Card_Category","Income_Category",
        │                    "Marital_Status","Education_Level","Gender"),
        │                  one_hot = TRUE)
        │
        ├── Step 6: Correlation Analysis
        │       cor(df_num, use = "pairwise.complete.obs")
        │       → pivot_longer → ggplot heatmap
        │           (steelblue = negative, pink = positive, green = zero)
        │
        └── Step 7: Class Imbalance Correction
                step_smote(Attrition_Flag)
                → synthetic minority oversampling on training data only
                → target class ratio corrected before model sees data
        │
        ▼
prep(recipe.df) → bake(recipe.df_prep, df_train)
        │
        ▼
vfold_cv(df_train, v = 5, strata = "Attrition_Flag")
        → stratified 5-fold CV — preserves 16.1% minority class in each fold
        │
        ▼
boost_tree(trees=tune(), learn_rate=tune(), tree_depth=tune()) %>%
  set_mode("classification") %>%
  set_engine("xgboost",
    tree_method = "hist",     ← histogram-based split finding
    device      = "cuda",     ← GPU acceleration
    max_bin     = 64,         ← histogram bin count
    nthread     = 10          ← CPU thread count for data loading
  )
        │
        ▼
workflow() %>% add_recipe(recipe.df) %>% add_model(xgboost.model)
        │
        ▼
Hyperparameter Search Space (extract_parameter_set_dials):
        trees:       range c(100, 500)
        learn_rate:  range c(0.0001, 0.1)
        tree_depth:  range c(3, 8)
        │
        ├── Stage 1: tune_grid(grid = 10, metrics = roc_auc + ppv + npv + accuracy)
        │       → 10 random configurations over k.fold
        │       → establishes exploration baseline (initial.grid)
        │
        └── Stage 2: tune_bayes(iter = 100, initial = initial.grid)
                → 100 Bayesian optimisation iterations
                → acquires next candidate by maximising expected improvement
                → converges on optimal region from Stage 1 baseline
        │
        ▼
select_best(metric = "roc_auc")
        → best.parameters → finalize_workflow(best.parameters_metrics)
        │
        ▼
last_fit(df.split, metrics = metric_set(
    accuracy, bal_accuracy, roc_auc,
    ppv, npv, f_meas, sens, spec, mcc))
        │
        ├── collect_metrics() → metrics table + bar chart
        ├── collect_predictions() → conf_mat() → autoplot(type="heatmap")
        └── extract_workflow() → saveRDS("final_model.rds")
```

---

## Key Technical Details

### 1. Two-Stage Hyperparameter Search

The pipeline uses a **warm-started Bayesian optimisation** strategy rather than a flat random or grid search:

- **Stage 1** — `tune_grid(grid=10)`: samples 10 random parameter combinations from the defined ranges, evaluating each on all 5 CV folds. This generates the `initial.grid` landscape.
- **Stage 2** — `tune_bayes(iter=100, initial=initial.grid)`: runs 100 Bayesian iterations, each using a surrogate model (Gaussian process) fitted to all previous results to select the most promising next candidate via expected improvement.

This is significantly more efficient than 110-point random search: the first 10 points explore; the subsequent 100 exploit the learned response surface. Best model selected by `roc_auc` — appropriate for imbalanced classification since it is threshold-independent.

### 2. IQR Outlier Detection + Winsorisation

```r
lower = Q1 − 1.5 × IQR
upper = Q3 + 1.5 × IQR
n_outliers = sum(col < lower | col > upper)
```

Rather than dropping outliers (which loses rows) or capping them manually (which is dataset-specific), `step_hai_winsorized_truncate(fraction = 0.05)` from `healthyR.ai` clips values at the 5th/95th percentile **inside the recipe** — meaning the same transformation is applied consistently to train, validation, and test data using percentiles computed from training data only. No data leakage.

### 3. Semantic Categorical Collapsing Before Encoding

Standard pipelines apply `step_dummy()` directly to raw categories. Here, semantically similar levels are **merged first** using `fct_collapse()` inside `step_mutate()`:

- Income bands "$40K-$60K", "$60K-$80K", "$80K-$120K", and "Unknown" → single level — reduces cardinality and prevents XGBoost from treating "Unknown" as an uninformative but separate dummy variable
- `step_novel()` is applied before each collapse to handle categories unseen at training time — preventing runtime failures during scoring

This reduces the one-hot dimension and improves model interpretability.

### 4. SMOTE on Training Data Only (Inside Recipe)

`step_smote(Attrition_Flag)` is **inside the recipe**, not applied to the raw data. This means:
- SMOTE is applied only to training folds during CV — test folds always remain original, unaugmented data
- SMOTE runs after normalization and encoding, so synthetic samples are generated in feature space, not raw category space
- Class ratio shifts from 83.9% / 16.1% toward 50/50, giving XGBoost equal gradient signal from both classes

### 5. GPU-Accelerated XGBoost

```r
set_engine("xgboost",
  tree_method = "hist",
  device      = "cuda",
  max_bin     = 64,
  nthread     = 10
)
```

- `tree_method = "hist"`: histogram-based approximate split finding — O(n·bins) instead of O(n·log n); required for GPU training
- `device = "cuda"`: offloads split evaluation and gradient computation to GPU
- `max_bin = 64`: controls histogram resolution — 64 bins is a pragmatic balance between granularity and GPU memory bandwidth
- `nthread = 10`: CPU threads for data I/O and preprocessing while GPU trains

See [GPU setup for XGBoost in R on Windows](https://stackoverflow.com/questions/77931563/how-to-use-xgboost-in-r-with-gpu) for installation instructions.

### 6. Stratified Cross-Validation

```r
vfold_cv(df_train, v = 5, strata = "Attrition_Flag")
```

Without stratification on an 83.9/16.1 split, a fold could contain fewer than expected minority-class rows purely by random chance, producing misleading validation metrics. `strata = "Attrition_Flag"` enforces that each fold preserves the training-set class ratio.

### 7. Comprehensive Final Evaluation Metric Set

```r
metric_set(accuracy, bal_accuracy, roc_auc,
           ppv, npv, f_meas, sens, spec, mcc)
```

- `mcc` (Matthews Correlation Coefficient) — the single most informative metric for imbalanced binary classification; takes into account all four confusion matrix cells
- `bal_accuracy` — arithmetic mean of sensitivity and specificity; unaffected by class ratio
- `ppv` / `npv` — precision on positive and negative classes respectively; reported separately to reveal asymmetric error costs

---

## Visualisations in Report

The rendered `BankingChurn.html` contains:

- **Attrition_Flag distribution** — proportional bar chart showing 83.9% vs 16.1% split
- **NA values per column** — horizontal bar chart; zero NA in this dataset (one column flagged for removal)
- **Outlier counts per feature** — bar chart of IQR-flagged values per numeric column
- **Feature scale distribution** — log₁₀(|Mean − SD|) per feature; motivates normalisation step
- **Pairwise correlation heatmap** — full numeric feature × feature matrix with diverging steelblue–green–pink colour scale
- **Model performance metrics** — horizontal bar chart of all 9 evaluation metrics
- **Confusion matrix heatmap** — `autoplot(conf_mat, type = "heatmap")` on test-set predictions

---

## Project Structure

```
Banking-Customer-Churn-Prediction/
├── BankingChurn.Rmd        # Full R Markdown pipeline — single source of truth
├── BankingChurn.html       # Rendered report with all code, outputs, and plots
├── BankingChurn.tex        # LaTeX intermediate (PDF output)
├── BankingChurn.log        # Knit log
├── BankChurners.csv        # Raw dataset — 10,127 rows × 22 columns
├── final_model.rds         # Saved tidymodels workflow (recipe + XGBoost model)
├── xgboost_r_gpu.tar.gz    # GPU-enabled XGBoost build for Windows R
└── .Rhistory
```

---

## Reproducing the Report

```r
# Install packages (handled automatically in Setup chunk)
# Knit the report in RStudio, or run from R console:
rmarkdown::render("BankingChurn.Rmd", output_format = "html_document")
```

Update `file_path` and `model_file.path` in the `.Rmd` to your local paths before running:

```r
file_path = "path/to/BankChurners.csv"
model_file.path = "path/to/save/"
```

To load and score with the saved model:

```r
library(tidymodels)
final_model = readRDS("final_model.rds")
predictions = predict(final_model, new_data = new_customer_df)
```

---

## R Package Dependencies

| Package | Role |
|---|---|
| `tidymodels` | Unified ML framework — rsample, recipes, parsnip, workflows, tune, yardstick |
| `xgboost` | Gradient boosting (GPU via CUDA) |
| `bonsai` | tidymodels engine bindings for XGBoost, LightGBM |
| `themis` | Resampling methods for imbalanced data — SMOTE |
| `healthyR.ai` | `step_hai_winsorized_truncate()` for recipe-level winsorisation |
| `tidyverse` | Data manipulation, ggplot2 visualisation |
| `DataExplorer` | Automated EDA utilities |
| `visdat` | Visual missing-data inspection |
| `forcats` | `fct_collapse()` for categorical recoding |
| `rsample` | `initial_split()`, `vfold_cv()` |
| `multcomp` | Post-hoc statistical comparisons |
| `jsonlite` | JSON serialisation |
| `data.table` | Fast data I/O |

---

## Sector Applications

| Sector | Application |
|---|---|
| **Finance** | Direct application — customer retention analytics, churn risk scoring, CLV prioritisation; pipeline transferable to credit default, fraud, and propensity-to-buy models |
| **Healthcare & Omics Research** | Patient dropout / non-adherence prediction; classification of at-risk patients from longitudinal clinical records; SMOTE + XGBoost applicable to imbalanced diagnostic datasets |
| **Technology / ML Engineering** | Template for reproducible tidymodels pipelines; recipe-as-preprocessing-graph pattern; Bayesian hyperparameter search with warm starts; RDS model serialisation for production scoring |

---

## Author

**Thiruvel Andagurunathan Pandian** — MSc Data Science, University of Bristol  
Building interpretable, production-ready ML pipelines across finance, healthcare, and research domains.  
📍 Bristol, UK · **Eligible for Skilled Worker Visa sponsorship** · Open to UK roles

[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://linkedin.com/in/Thiruvel-AP)
[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/Thiruvel-AP)

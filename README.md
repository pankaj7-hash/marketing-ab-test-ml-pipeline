# 📊 Marketing A/B Test — End-to-End ML Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3+-orange?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google%20Colab-Ready-yellow?style=for-the-badge&logo=googlecolab&logoColor=white"/>
  <img src="https://img.shields.io/badge/ROC--AUC-0.8498-success?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge"/>
</p>

<p align="center">
  A production-ready, 20-section data science notebook that takes raw marketing A/B test data from <b>588K rows</b> all the way to a <b>saved, reloadable Random Forest classifier</b> with ROC-AUC of <b>0.8498</b>.
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Pipeline Sections](#-pipeline-sections)
- [Key Results](#-key-results)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Notebook Walkthrough](#-notebook-walkthrough)
- [Model Performance](#-model-performance)
- [Key Findings & Business Insights](#-key-findings--business-insights)
- [Engineering Highlights](#-engineering-highlights)
- [Next Steps](#-next-steps)
- [License](#-license)

---

## 🧭 Overview

This project presents a **complete, end-to-end machine learning pipeline** built on a real-world marketing A/B test dataset. The central business question is:

> **Does showing ads actually drive user conversions — and can we predict who will convert?**

The pipeline covers everything from raw data ingestion and statistical hypothesis testing to model training, hyperparameter tuning, evaluation, and model persistence — all in a single, well-structured Google Colab notebook.

---

## 📂 Dataset

| Property | Detail |
|---|---|
| **File** | `marketing_AB.csv` |
| **Rows** | ~588,000 user records |
| **Target** | `converted` — binary (0 = not converted, 1 = converted) |
| **Class imbalance** | ~97% not converted / ~3% converted |
| **Source** | Marketing A/B test campaign data |

### Columns

| Column | Type | Description |
|---|---|---|
| `user id` | Integer | Unique user identifier (dropped before modelling) |
| `test group` | Categorical | `ad` — saw advertisement; `psa` — saw public service announcement |
| `converted` | Boolean → Integer | Whether the user converted (target variable) |
| `total ads` | Numerical | Total number of ads seen by the user |
| `most ads day` | Categorical | Day of the week when the user saw the most ads |
| `most ads hour` | Numerical (0–23) | Hour of the day when the user saw the most ads |

---

## 🗂 Pipeline Sections

| # | Section | Description |
|---|---|---|
| 1 | 🔧 Setup & Install | Library installation and global imports |
| 2 | 📂 Load & Inspect | Upload CSV, inspect shape, dtypes, head |
| 3 | 🧹 Data Cleaning | Column renaming, missing values, deduplication, outlier capping (3×IQR), bool→int conversion |
| 4 | 🔍 Column Exploration | Identify numerical/categorical columns, value counts, descriptive statistics |
| 5 | 📈 Univariate Analysis | Histograms, KDE, log-scale plots, bar charts, target distribution |
| 6 | 🔗 Bivariate Analysis | Conversion rate by group/day/hour, boxplots, histograms by class, crosstabs |
| 7 | 🌐 Multivariate Analysis | Day × Group heatmap, hour × group line chart, ads-bin conversion rate bars |
| 8 | 🌡️ Correlation Heatmap | Lower-triangle correlation matrix with encoded features |
| 9 | 🧪 Statistical Tests | Chi-squared, Mann-Whitney U, Point-Biserial — formal A/B significance testing |
| 10 | ⚙️ Feature Engineering | Log transform, hour buckets, weekend flag, ads intensity bins |
| 11 | 🔢 Encoding | Label, ordinal, and binary encoding of all categorical columns |
| 12 | ✂️ Train/Test Split | 80/20 stratified split on original data |
| 13 | ⚖️ SMOTE | Oversample minority class to 30% of majority on training data |
| 14 | 🔀 Train/Validation Split | Further split resampled training data 80/20 |
| 15 | 🌲 Model Training | Logistic Regression (baseline) + Random Forest (200 trees) + 5-fold cross-validation |
| 16 | 🟥 Confusion Matrix | Confusion matrix, ROC Curve, Precision-Recall Curve |
| 17 | 🏆 Feature Importance | Gini importance + Permutation importance (ROC-AUC based) |
| 18 | 🎛️ Hyperparameter Tuning | Optimized `RandomizedSearchCV` on stratified subsample — search in minutes not hours |
| 19 | 🎯 Final Evaluation | Full classification report, ROC-AUC, F1, Avg Precision on held-out test set |
| 20 | 💾 Save & Reload | `joblib` model + metadata save/reload + Colab download |

---

## 📊 Key Results

| Metric | Score |
|---|---|
| **Test ROC-AUC** | **0.8498** |
| **Test Avg Precision** | — |
| **F1 Score (macro)** | — |
| **CV ROC-AUC (5-fold)** | — |

> Exact CV and F1 values appear in notebook output after execution.

---

## 🛠 Tech Stack

| Category | Libraries |
|---|---|
| **Data** | `pandas`, `numpy` |
| **Visualisation** | `matplotlib`, `seaborn` |
| **Statistics** | `scipy` (chi2, Mann-Whitney U, point-biserial) |
| **ML** | `scikit-learn` (RF, LR, CV, metrics, encoding) |
| **Imbalance** | `imbalanced-learn` (SMOTE) |
| **Persistence** | `joblib` |
| **Environment** | Google Colab (Python 3.10+) |

---

## 📁 Project Structure

```
marketing-ab-test-ml-pipeline/
│
├── A_B_test.ipynb              # Main notebook (Google Colab)
├── marketing_AB.csv            # Dataset (upload manually in Colab)
├── best_rf_model.joblib        # Saved trained model (generated after run)
├── model_metadata.joblib       # Feature names, encodings, best params, AUC
├── README.md                   # This file
└── LICENSE                     # MIT License
```

---

## 🚀 Getting Started

### Option 1 — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `A_B_test.ipynb` via **File → Upload notebook**
3. Run **Section 1** to install dependencies
4. Run **Section 2** — the file picker will prompt you to upload `marketing_AB.csv`
5. Run all remaining cells sequentially

### Option 2 — Local Jupyter

```bash
# 1. Clone the repository
git clone https://github.com/your-username/marketing-ab-test-ml-pipeline.git
cd marketing-ab-test-ml-pipeline

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn joblib jupyter

# 4. Launch Jupyter
jupyter notebook A_B_test.ipynb
```

> **Note:** In local mode, replace the `files.upload()` cell in Section 2 with `pd.read_csv('marketing_AB.csv', index_col=0)`.

### Requirements

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
scipy>=1.9
scikit-learn>=1.3
imbalanced-learn>=0.11
joblib>=1.2
```

---

## 📖 Notebook Walkthrough

### Data Cleaning (Section 3)
- Column names normalised to `snake_case`
- Zero missing values confirmed
- Duplicate rows removed
- `total_ads` outliers capped at **Q3 + 3×IQR** (soft cap — preserves class distribution)
- `converted` cast from bool → int

### Feature Engineering (Section 10)
Four new features created from existing columns:

| New Feature | Source | Logic |
|---|---|---|
| `log_total_ads` | `total_ads` | `log(1 + total_ads)` — corrects right skew |
| `hour_bucket` | `most_ads_hour` | morning / afternoon / evening / night |
| `is_weekend` | `most_ads_day` | 1 if Saturday/Sunday, else 0 |
| `ads_intensity` | `total_ads` | low (0–50) / medium (51–200) / high (200+) |

### Handling Class Imbalance (Section 13)
The dataset is severely imbalanced (~97% non-converted). **SMOTE** (Synthetic Minority Oversampling Technique) is applied to the training set only, bringing the minority class to 30% of the majority — preventing data leakage into the test set.

### Statistical A/B Testing (Section 9)

| Test | Variables | Result |
|---|---|---|
| Chi-squared | `test_group` × `converted` | **Significant** (p < 0.05) |
| Mann-Whitney U | `total_ads` by conversion | **Significant** (p < 0.05) |
| Point-Biserial | `total_ads` ↔ `converted` | Positive correlation |
| Chi-squared | `most_ads_day` × `converted` | **Significant** (p < 0.05) |

### Optimized Hyperparameter Tuning (Section 18)

The naive approach (RandomizedSearchCV on 700K SMOTE rows) takes **1+ hour**. The optimized 3-step strategy:

```
Step 1 → Stratified 10% subsample (~70K rows, balanced classes)
Step 2 → RandomizedSearchCV: 15 iters × 3-fold CV = 45 fits on 70K rows (~3–5 min)
Step 3 → Refit best params on full 700K rows with n_estimators=200 (single fit)
```

**Result: Same AUC, 95% less search time.**

---

## 📈 Model Performance

### Model Comparison

| Model | Dataset | ROC-AUC |
|---|---|---|
| Logistic Regression (baseline) | Validation | — |
| Random Forest (default, 200 trees) | Validation | — |
| **Random Forest (tuned)** | **Test** | **0.8498** |

> Values populated after notebook execution.

### Feature Importance (Gini — top features)

```
log_total_ads       ████████████████████  (highest)
total_ads           ██████████████████
total_ads_capped    █████████████████
most_ads_hour       ████████████
day_enc             ████████
test_group_enc      █████
hour_bucket_enc     ████
ads_intensity_enc   ███
is_weekend          ██
```

> Ad exposure volume is by far the strongest predictor of conversion.

---

## 💡 Key Findings & Business Insights

**1. Ads work — statistically confirmed**
The Chi-squared test confirms the ad group converts at a significantly higher rate than the PSA group (p < 0.05).

**2. Volume matters most**
`log_total_ads` is the strongest predictor. Users who see more ads are substantially more likely to convert — but with diminishing returns (hence the log transform).

**3. Timing drives conversion**
Evening hours show the highest conversion rates across both groups. Campaigns should be weighted toward evening slots.

**4. Day of week has impact**
Certain weekdays outperform others. The heatmap (Section 7) reveals the best day × group combinations for ad delivery.

**5. Class imbalance is severe**
Only ~3% of users convert. Ignoring this leads to a model that predicts "never convert" with 97% accuracy — useless for business. SMOTE corrects for this.

---

## ⚡ Engineering Highlights

- **Dark-themed visualizations** — custom matplotlib rcParams with a consistent colour palette across all 20+ plots
- **Stratified sampling everywhere** — train/test split, SMOTE application, tuning subsample all preserve class ratios
- **No data leakage** — SMOTE applied only on training data; scaler fit only on training data
- **Permutation importance** alongside Gini importance for a more robust feature ranking
- **Reusable `plot_confusion_matrix` function** — called for both validation and test sets
- **Metadata persistence** — encoding maps, feature list, best params, and AUC all saved alongside the model for reproducible inference

---

## 🔮 Next Steps

- [ ] Try **XGBoost / LightGBM** — likely to improve AUC further with less tuning
- [ ] **SHAP values** for model explainability and individual prediction interpretation
- [ ] **Threshold tuning** — optimise decision threshold based on business cost of false positives vs false negatives
- [ ] **Uplift modelling** — measure true incremental causal impact of ads (beyond correlation)
- [ ] **FastAPI inference endpoint** — wrap the saved model into a REST API for production serving
- [ ] **Streamlit dashboard** — interactive EDA + prediction interface

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙋 About

Built by **Pankaj** as part of an AI/ML Engineering portfolio targeting production-ready data science projects.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/your-username)

---

<p align="center">
  ⭐ If this project helped you, consider giving it a star!
</p>

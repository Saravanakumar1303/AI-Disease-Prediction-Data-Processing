# 🏥 AI-Based Disease Prediction System

> A Machine Learning project that predicts the likelihood of **Diabetes**, **Heart Disease**, and **Liver Disease** using real-world clinical datasets from Kaggle. This module covers data preprocessing and model training — part of a larger full-stack AI system with a Python API backend.

---
## 📁 Project Structure
```
├── raw_datasets/
│   ├── diabetes.csv
│   ├── heart.csv
│   └── indian_liver_patient.csv
│
├── clean_datasets/
│   ├── diabetes_clean.csv
│   ├── heart_clean.csv
│   └── liver_clean.csv
│
├── diabetes_clean.py
├── heart_clean.py
└── liver_clean.py
```
## 🧬 Datasets Used

| Dataset | Source | Rows | Columns | Target Column |
|---|---|---|---|---|
| Pima Indians Diabetes | Kaggle | 768 | 9 | `Outcome` (0/1) |
| Heart Disease UCI | Kaggle | 303 | 14 | `target` (0/1) |
| Indian Liver Patient | Kaggle | 583 | 11 | `Dataset` → `target` (0/1) |

---

## 🧹 Data Cleaning — Issues Found & Fixed

### 1. 🩸 Diabetes Dataset (`diabetes.csv`)

| # | Issue | Details | Fix Applied |
|---|---|---|---|
| 1 | **Hidden Missing Values** | Medically impossible `0` values — `Glucose` (0.7%), `BloodPressure` (4.6%), `SkinThickness` (29.6%), `Insulin` (48.7%), `BMI` (1.4%) | Replaced `0` → `NaN`, then filled with **column median** (median chosen over mean due to skewed distribution) |
| 2 | **Class Imbalance** | `Outcome=0`: 500 samples vs `Outcome=1`: 268 samples (~65/35 split) | Used `class_weight='balanced'` in RandomForestClassifier |
| 3 | **Outliers** | Extreme values in `Insulin`, `SkinThickness`, `BMI`, `DiabetesPedigreeFunction` | Capped using **IQR method** (clip at Q1−1.5×IQR and Q3+1.5×IQR) |

---

### 2. ❤️ Heart Disease Dataset (`heart.csv`)

| # | Issue | Details | Fix Applied |
|---|---|---|---|
| 1 | **Invalid `thal` values** | 2 rows with `thal=0` — medically invalid (valid values: 1, 2, 3 only) | Replaced `0` with **column mode** |
| 2 | **Invalid `ca` values** | 5 rows with `ca=4` — should be 0–3 only | Replaced `4` → `NaN` → filled with **column mode** |
| 3 | **Duplicate Rows** | 1 duplicate row found | Removed using `drop_duplicates()` |
| 4 | **Cholesterol Outlier** | Some values above 400 mg/dL (clinically extreme) | Capped at **400 mg/dL** using `clip()` |
| 5 | **Cryptic Column Names** | `cp`, `trestbps`, `thalach`, `exang`, `oldpeak` — not readable | Renamed to full readable names (`chest_pain_type`, `resting_bp`, `max_heart_rate`, etc.) |

---

### 3. 🫀 Indian Liver Patient Dataset (`indian_liver_patient.csv`)

| # | Issue | Details | Fix Applied |
|---|---|---|---|
| 1 | **Missing Values** | 4 missing values in `Albumin_and_Globulin_Ratio` column | Filled with **column median** |
| 2 | **Gender Encoding** | `Gender` column had `Male`/`Female` strings | Label encoded → `1`/`0` using `LabelEncoder` |
| 3 | **Duplicate Rows** | 13 duplicate rows found | Removed using `drop_duplicates()` |
| 4 | **Severe Outliers** | `Alamine_Aminotransferase` max=2000, `Aspartate_Aminotransferase` max=4929 (75th percentile only ~60 and ~87) | Capped all numeric columns using **IQR method** |
| 5 | **Confusing Column Names** | `TB`, `DB`, `Alkphos`, `Sgpt`, `Sgot`, `TP`, `ALB`, `A/G Ratio` | Renamed to full medical terms |
| 6 | **Target Column Issues** | Column named `Dataset`, values `1=Disease`, `2=No Disease` — confusing | Renamed to `target`, recoded `2 → 0` (standard binary: `1=Disease`, `0=No Disease`) |
| 7 | **Class Imbalance** | 71.4% liver patients vs 28.6% healthy | Noted for model evaluation; addressed via `stratify=y` in train-test split |

### 4. 🫘 Chronic Kidney Dataset (`kidney_clean.csv`)

| # | Issue | Details | Fix Applied |
|---|---|---|---|
| 1 | **Missing Values** | `rc` 32.5%, `rbc` 38%, `wc` 26.25%, `pot` 22%, `sod` 21.75%, `pcv` 17.5% — 24 columns-ல் missing data | Numeric columns → **Median**, Categorical columns → **Mode** |
| 2 | **Wrong Dtype** | `pcv`, `wc`, `rc` — numeric values ஆனா `string` type-ல store ஆகி இருக்கு | `str.strip()` → `pd.to_numeric(errors='coerce')` |
| 3 | **Dirty String Values** | `dm` column-ல `' yes'`, `'\tno'`, `'\tyes'` — leading spaces & tab characters | `.str.strip().str.lower()` applied on all 10 categorical columns |
| 4 | **Target Label Dirt** | `classification` column-ல `'ckd\t'` — trailing tab character | `.str.strip()` → map `'ckd'→1`, `'notckd'→0` |
| 5 | **Unnecessary Column** | `id` column — ML-க்கு useless | `df.drop(columns=['id'])` |
| 6 | **Label Encoding** | `rbc`, `pc`, `pcc`, `ba`, `htn`, `dm`, `cad`, `appet`, `pe`, `ane` — 10 categorical columns | `LabelEncoder` → all converted to int64 |
| 7 | **Class Imbalance** | 62.5% CKD patients vs 37.5% healthy | `stratify=y` in train-test split |
---

## ⚙️ Preprocessing Pipeline (All Datasets)

```
Raw CSV → Handle Missing Values → Fix Invalid Values → Remove Duplicates
       → Treat Outliers (IQR) → Encode Categorical → Rename Columns
       → Save Clean CSV → Train/Test Split (80/20, stratified)
       → StandardScaler → Model Training
```


## 🤖 Model Training

### Models Trained

| Model | Why Used |
|---|---|
| **Logistic Regression** | Simple, interpretable baseline for binary classification |
| **Random Forest** | Handles non-linearity and feature interactions well |
| **Gradient Boosting** | Strong performance on small-medium tabular datasets |
| **SVM (RBF Kernel)** | Effective in high-dimensional spaces with scaling |
| **KNN (k=5)** | Distance-based, good for local pattern detection |

### Training Configuration

| Parameter | Value |
|---|---|
| Test Size | 20% |
| Train Size | 80% |
| Split Strategy | Stratified (preserves class ratio) |
| Feature Scaling | StandardScaler (zero mean, unit variance) |
| Random State | 42 (reproducibility) |

---

## 📊 Final Results Summary

| Disease | Best Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| 🩸 Diabetes | Random Forest | ~76% | — | — | — |
| ❤️ Heart Disease | *(run to get results)* | — | — | — | — |
| 🫀 Liver Disease | *(run to get results)* | — | — | — | — |

> ⚠️ *Run the respective `.py` files to get exact metrics. The best model is auto-selected and printed at the end of each script.*

---

## 🚀 How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn

# Run each disease training script
diabetes_clean.py
heart_clean.py
liver_clean.py
kidney_clean.py
```

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Pandas | Data loading & cleaning |
| NumPy | Numerical operations |
| Scikit-learn | ML models, scaling, evaluation |
| Kaggle | Dataset source |

---

## 👤 My Role

This project is part of a larger **AI-Based Disease Prediction System** built by a team. My contribution covers the **ML/Data Science module**:
- Downloaded and explored datasets from Kaggle
- Identified and fixed all data quality issues
- Built and compared multiple ML models for each disease
- Selected the best performing model based on accuracy, precision, recall, and F1 score
- Saved clean datasets and prepared them for API integration

The **Python API development** (Django/FastAPI) integrates these trained models for real-time predictions.

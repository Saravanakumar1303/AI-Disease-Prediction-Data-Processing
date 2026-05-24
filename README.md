# 🏥 AI-Based Early Disease Prediction System

> An AI-powered early disease prediction system built for UK healthcare clinics — detecting diseases at early stages using real patient data and machine learning.

📌 Project Overview
Many diseases in the UK are diagnosed too late — diabetes, heart disease, kidney disease, and more. This project builds an AI system that:

Predicts diseases at early stages using patient data
Supports doctors in making faster, data-driven decisions
Enables preventive healthcare through risk scoring

Built for: UK Healthcare Clinics
Role: Python Development 
Timeline: 80-Day Project

| Disease | Dataset | Rows | Columns | Source |
|---------|---------|------|---------|--------|
| Diabetes | Pima Indians Diabetes | 768 | 9 | Kaggle |
| Heart Disease | Heart Disease UCI | 303 | 14 | Kaggle |
| Kidney Disease | Chronic Kidney Disease | 400 | 26 | Kaggle |
| Liver Disease | Indian Liver Patient | 583 | 11 | Kaggle |
| General Symptoms | Disease & Symptoms | 4920 | 18 | Kaggle |

## 🧹 Week 2 — Data Preprocessing Progress

### ✅ Day 1 — Diabetes Dataset (`diabetes_preprocessing.py`)

**Issues Found:**

| Issue | Detail |
|-------|--------|
| Hidden missing values | Glucose, BP, Insulin, BMI, SkinThickness had medically impossible 0s |
| Outliers | Insulin max=846, SkinThickness max=99 |
| Class imbalance | No Diabetes: 500, Diabetes: 268 (65/35 split) |
| Feature scale difference | Age (21–81) vs Insulin (15–300+) on different scales |

**Fixes Applied:**

```python
# 1. Replace medically invalid zeros with NaN, fill with median
zero_not_allowed = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[zero_not_allowed] = df[zero_not_allowed].replace(0, np.nan)
df[zero_not_allowed] = df[zero_not_allowed].fillna(df[zero_not_allowed].median())

# 2. IQR-based outlier capping (Winsorization)
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df[column] = df[column].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
    return df

# 3. class_weight='balanced' — no fake data (SMOTE avoided)
# 4. stratify=y — fair train/test split
# 5. StandardScaler — normalize all features
```

**Model Result:**
```
Model    : Random Forest Classifier
Accuracy : 75.32%
Data     : Real patients only — no synthetic data
```

---

### ⏳ Day 2 — Heart Disease Dataset (`heart_preprocessing.py`)
> Coming soon

---

### ⏳ Day 3 — Chronic Kidney Disease Dataset (`ckd_preprocessing.py`)
> Coming soon

---

### ⏳ Day 4 — Indian Liver Patient Dataset (`liver_preprocessing.py`)
> Coming soon

---

### ⏳ Day 5 — Disease & Symptoms Dataset (`symptoms_preprocessing.py`)
> Coming soon

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/AI-Disease-Prediction-System.git
cd AI-Disease-Prediction-System
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing
```bash
# Diabetes
diabetes_clean.py

# Heart (coming soon)
heart_clean.py
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
```

Install all at once:
```bash
pip install pandas numpy scikit-learn
```
## 🧠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.x |
| Data Processing | pandas, numpy |
| ML Models | scikit-learn |
| Backend | Flask / Django (planned) |
| Frontend | React / Streamlit (planned) |
| Database | PostgreSQL (planned) |
| Cloud | AWS / Azure (planned) |

---

## 📝 License

This project is for educational and demonstration purposes.  
Dataset credits: Kaggle contributors.

---



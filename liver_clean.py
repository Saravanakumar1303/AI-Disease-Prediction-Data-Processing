import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
#  STEP 1 : LOAD DATASET
# ============================================================
print("=" * 60)
print("   LIVER DISEASE - CLEANING + TRAINING REPORT")
print("=" * 60)

# Load dataset
# Change file name if needed
# Example: indian_liver_patient.csv

df = pd.read_csv('raw_datasets/indian_liver_patient.csv')

print(f"\n📂 Original Dataset Shape : {df.shape[0]} rows × {df.shape[1]} columns")

# ============================================================
#  STEP 2 : DATA CLEANING
# ============================================================
print("\n" + "=" * 60)
print("   STEP 2 : DATA CLEANING")
print("=" * 60)

# ── Fix 1: Missing Values ───────────────────────────────────
missing_before = df.isnull().sum().sum()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

missing_after = df.isnull().sum().sum()

print(f"\n✅ Missing Values Fixed : {missing_before} → {missing_after}")

# ── Fix 2: Gender Encoding ──────────────────────────────────
if 'Gender' in df.columns:
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    print("✅ Gender Encoded       : Male/Female → 1/0")

# ── Fix 3: Duplicate Rows ───────────────────────────────────
dup_count = df.duplicated().sum()
df = df.drop_duplicates()

print(f"✅ Duplicates Fixed    : {dup_count} row(s) removed")

# ── Fix 4: Outlier Treatment ────────────────────────────────
# Cap numerical columns using IQR method

numeric_cols = df.select_dtypes(include=np.number).columns
outlier_count = 0

for col in numeric_cols:
    if col.lower() not in ['dataset', 'target']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        before = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_count += before

        df[col] = np.clip(df[col], lower, upper)

print(f"✅ Outliers Fixed      : {outlier_count} values capped using IQR")

# ── Fix 5: Rename Columns ───────────────────────────────────
df = df.rename(columns={
    'TB'  : 'total_bilirubin',
    'DB'  : 'direct_bilirubin',
    'Alkphos' : 'alkaline_phosphotase',
    'Sgpt' : 'alamine_aminotransferase',
    'Sgot' : 'aspartate_aminotransferase',
    'TP'   : 'total_proteins',
    'ALB'  : 'albumin',
    'A/G Ratio' : 'albumin_globulin_ratio',
    'Dataset' : 'target'
})

print("✅ Columns Renamed     : Medical names converted to readable format")

# ── Fix 6: Target Column Standardization ───────────────────
# Dataset column usually contains:
# 1 = Liver Disease
# 2 = No Liver Disease

if 'target' in df.columns:
    df['target'] = df['target'].replace({2: 0})
    print("✅ Target Standardized : 1 = Disease, 0 = No Disease")

# ============================================================
#  SAVE CLEAN DATASET
# ============================================================
os.makedirs('clean_datasets', exist_ok=True)

clean_path = 'clean_datasets/liver_clean.csv'
df.to_csv(clean_path, index=False)

print(f"\n💾 Clean Dataset Saved : {clean_path}")
print(f"📊 Clean Dataset Shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"🔍 Missing Values      : {df.isnull().sum().sum()} ✅")
print(f"🔍 Duplicate Rows      : {df.duplicated().sum()} ✅")

# ============================================================
#  STEP 3 : PREPARE DATA FOR TRAINING
# ============================================================
print("\n" + "=" * 60)
print("   STEP 3 : PREPARE DATA FOR TRAINING")
print("=" * 60)

# Features & Target
X = df.drop('target', axis=1)
y = df['target']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📌 Features (X)     : {X.shape[1]} columns")
print(f"📌 Target (y)       : 'target' column")
print(f"📌 Train Size       : {X_train.shape[0]} rows (80%)")
print(f"📌 Test Size        : {X_test.shape[0]} rows (20%)")
print(f"📌 Scaling          : StandardScaler applied ✅")

# ============================================================
#  STEP 4 : TRAIN MULTIPLE MODELS
# ============================================================
print("\n" + "=" * 60)
print("   STEP 4 : MODEL TRAINING & ACCURACY")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = {}
best_model_name = None
best_accuracy = 0

print()
print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for name, model in models.items():

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        'model': model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'y_pred': y_pred
    }

    flag = " ⭐" if acc > best_accuracy else ""

    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name

    print(f"  {name:<25} {acc*100:>9.2f}% {prec*100:>9.2f}% {rec*100:>9.2f}% {f1*100:>9.2f}%{flag}")

# ============================================================
#  STEP 5 : BEST MODEL REPORT
# ============================================================
print("\n" + "=" * 60)
print(f"   STEP 5 : BEST MODEL → {best_model_name}")
print("=" * 60)

best = results[best_model_name]
y_pred_best = best['y_pred']

cm = confusion_matrix(y_test, y_pred_best)

print(f"\n🏆 Best Model       : {best_model_name}")
print(f"🎯 Accuracy         : {best['accuracy']*100:.2f}%")
print(f"🎯 Precision        : {best['precision']*100:.2f}%")
print(f"🎯 Recall           : {best['recall']*100:.2f}%")
print(f"🎯 F1 Score         : {best['f1']*100:.2f}%")

print(f"\n📋 Confusion Matrix :")
print(f"                  Predicted")
print(f"                  No Disease  Disease")
print(f"  Actual No Disease  {cm[0][0]:>5}     {cm[0][1]:>5}")
print(f"  Actual Disease     {cm[1][0]:>5}     {cm[1][1]:>5}")

print(f"\n📋 Classification Report :")
print(classification_report(
    y_test,
    y_pred_best,
    target_names=['No Disease', 'Liver Disease']
))

# ============================================================
#  FINAL SUMMARY
# ============================================================
print("=" * 60)
print("   ✅ FINAL SUMMARY")
print("=" * 60)

print(f"\n  Original Rows      : {df.shape[0] + dup_count}")
print(f"  Clean Rows         : {df.shape[0]}")
print(f"  Problems Fixed     : 6 (missing, encoding, duplicate, outlier, rename, target)")
print(f"  Best Model         : {best_model_name}")
print(f"  Best Accuracy      : {best_accuracy*100:.2f}%")
print(f"  Clean CSV Saved    : clean_datasets/indian_liver_patient_clean.csv ✅")
print("=" * 60)

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)

# ============================================================
#  STEP 1 : LOAD DATASET
# ============================================================
print("=" * 60)
print("   HEART DISEASE - CLEANING + TRAINING REPORT")
print("=" * 60)

df = pd.read_csv('raw_datasets/heart.csv')
print(f"\n📂 Original Dataset Shape : {df.shape[0]} rows × {df.shape[1]} columns")

# ============================================================
#  STEP 2 : DATA CLEANING
# ============================================================
print("\n" + "=" * 60)
print("   STEP 2 : DATA CLEANING")
print("=" * 60)

# ── Fix 1: thal = 0 → replace with mode ──────────────────────
thal_invalid = df[df['thal'] == 0].shape[0]
thal_mode = df[df['thal'] != 0]['thal'].mode()[0]
df['thal'] = df['thal'].replace(0, thal_mode)
print(f"\n✅ thal=0 Fixed     : {thal_invalid} rows → replaced with mode ({thal_mode})")

# ── Fix 2: ca = 4 → NaN → fill with mode ────────────────────
ca_invalid = df[df['ca'] == 4].shape[0]
df['ca'] = df['ca'].replace(4, np.nan)
ca_mode = df['ca'].mode()[0]
df['ca'] = df['ca'].fillna(ca_mode)
df['ca'] = df['ca'].astype(int)
print(f"✅ ca=4 Fixed       : {ca_invalid} rows → NaN → filled with mode ({int(ca_mode)})")

# ── Fix 3: Duplicate rows ────────────────────────────────────
dup_count = df.duplicated().sum()
df = df.drop_duplicates()
print(f"✅ Duplicates Fixed : {dup_count} row(s) removed")

# ── Fix 4: Cholesterol outlier ───────────────────────────────
chol_out = df[df['chol'] > 400].shape[0]
df['chol'] = df['chol'].clip(upper=400)
print(f"✅ Chol Outlier Fix : {chol_out} row(s) capped at 400 mg/dL")

# ── Fix 5: Rename columns ────────────────────────────────────
df = df.rename(columns={
    'cp'       : 'chest_pain_type',
    'trestbps' : 'resting_bp',
    'chol'     : 'cholesterol',
    'fbs'      : 'fasting_blood_sugar',
    'restecg'  : 'rest_ecg',
    'thalach'  : 'max_heart_rate',
    'exang'    : 'exercise_angina',
    'oldpeak'  : 'st_depression',
    'ca'       : 'num_vessels',
    'thal'     : 'thalassemia'
})
print(f"✅ Columns Renamed  : 10 columns → readable names")

# ── Save clean dataset ───────────────────────────────────────
os.makedirs('clean_datasets', exist_ok=True)
df.to_csv('clean_datasets/heart_clean.csv', index=False)
print(f"\n💾 Clean Dataset Saved : clean_datasets/heart_clean.csv")
print(f"📊 Clean Dataset Shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"🔍 Missing Values      : {df.isnull().sum().sum()} ✅")
print(f"🔍 Duplicate Rows      : {df.duplicated().sum()} ✅")

# ============================================================
#  STEP 3 : PREPARE DATA FOR TRAINING
# ============================================================
print("\n" + "=" * 60)
print("   STEP 3 : PREPARE DATA FOR TRAINING")
print("=" * 60)

X = df.drop('target', axis=1)   # Features
y = df['target']                 # Target (0 = No Disease, 1 = Disease)

# Train / Test Split — 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

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
    'Logistic Regression'    : LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest'          : RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting'      : GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM'                    : SVC(kernel='rbf', random_state=42),
    'KNN'                    : KNeighborsClassifier(n_neighbors=5),
}

results = {}
best_model_name = None
best_accuracy   = 0

print()
print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    results[name] = {
        'model'    : model,
        'accuracy' : acc,
        'precision': prec,
        'recall'   : rec,
        'f1'       : f1,
        'y_pred'   : y_pred
    }

    flag = " ⭐" if acc > best_accuracy else ""
    if acc > best_accuracy:
        best_accuracy   = acc
        best_model_name = name

    print(f"  {name:<25} {acc*100:>9.2f}% {prec*100:>9.2f}% {rec*100:>9.2f}% {f1*100:>9.2f}%{flag}")

# ============================================================
#  STEP 5 : BEST MODEL — DETAILED REPORT
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
print(classification_report(y_test, y_pred_best,
      target_names=['No Disease', 'Heart Disease']))

# ============================================================
#  FINAL SUMMARY
# ============================================================
print("=" * 60)
print("   ✅ FINAL SUMMARY")
print("=" * 60)
print(f"\n  Original Rows      : 303")
print(f"  Clean Rows         : {df.shape[0]}")
print(f"  Problems Fixed     : 5 (thal, ca, duplicate, outlier, rename)")
print(f"  Best Model         : {best_model_name}")
print(f"  Best Accuracy      : {best_accuracy*100:.2f}%")
print(f"  Clean CSV Saved    : clean_datasets/heart_clean.csv ✅")
print("=" * 60)
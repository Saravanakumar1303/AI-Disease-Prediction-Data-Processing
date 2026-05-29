import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

print("=" * 65)
print("  KIDNEY DISEASE — FULL PIPELINE (CLEAN + TRAIN + REPORT)")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# STEP 0: Load Dataset
# ─────────────────────────────────────────────────────────────
UPLOAD_PATHS = [
    "raw_datasets/ckd_dataset.csv",
    "/mnt/user-data/uploads/kidney_disease.csv",
    "/mnt/user-data/uploads/kidney.csv",
    "/mnt/user-data/uploads/chronic_kidney_disease.csv",
]

df = None
for path in UPLOAD_PATHS:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n✅ Loaded: {path}")
        break

if df is None:
    print("\n⚠️  No file found — generating UCI CKD demo dataset...\n")
    np.random.seed(42)
    n = 400

    def smear(s, pct):
        out = s.copy().astype(object)
        out.iloc[np.random.choice(len(out), int(len(out)*pct), replace=False)] = np.nan
        return out

    def buggy_str(vals, pct, tabs=5):
        s = pd.Series(vals.astype(str))
        for i in np.random.choice(n, tabs, replace=False):
            s.iloc[i] = "\t" + s.iloc[i]
        return smear(s, pct)

    def cat(choices, pct):
        return smear(pd.Series(np.random.choice(choices, n)), pct)

    dm_vals = []
    for v in np.random.choice(['yes','no'], n):
        r = np.random.random()
        dm_vals.append(' yes' if r<.15 else ('\tno' if r<.25 else ('\tyes' if r<.35 else v)))

    labels = [('ckd\t' if (v=='ckd' and np.random.random()<.30) else v)
              for v in np.random.choice(['ckd','notckd'], n, p=[0.625,0.375])]

    df = pd.DataFrame({
        'id'   : range(n),
        'age'  : smear(pd.Series(np.random.randint(2,90,n).astype(float)), 0.01),
        'bp'   : smear(pd.Series(np.random.choice([60.,70.,80.,90.,100.,110.],n)), 0.12),
        'sg'   : smear(pd.Series(np.random.choice([1.005,1.010,1.015,1.020,1.025],n)), 0.05),
        'al'   : smear(pd.Series(np.random.choice([0.,1.,2.,3.,4.,5.],n)), 0.05),
        'su'   : smear(pd.Series(np.random.choice([0.,1.,2.,3.,4.,5.],n)), 0.05),
        'rbc'  : cat(['normal','abnormal'], 0.20),
        'pc'   : cat(['normal','abnormal'], 0.05),
        'pcc'  : cat(['present','notpresent'], 0.01),
        'ba'   : cat(['present','notpresent'], 0.01),
        'bgr'  : smear(pd.Series(np.random.randint(70,490,n).astype(float)), 0.12),
        'bu'   : smear(pd.Series(np.random.randint(10,200,n).astype(float)), 0.10),
        'sc'   : smear(pd.Series(np.random.uniform(0.4,15.0,n).round(1)), 0.10),
        'sod'  : smear(pd.Series(np.random.randint(110,155,n).astype(float)), 0.22),
        'pot'  : smear(pd.Series(np.random.uniform(2.5,7.5,n).round(1)), 0.22),
        'hemo' : smear(pd.Series(np.random.uniform(3.1,17.8,n).round(1)), 0.10),
        'pcv'  : buggy_str(np.random.randint(16,54,n), 0.175),
        'wc'   : buggy_str(np.random.randint(2200,26400,n), 0.2625),
        'rc'   : buggy_str(np.random.uniform(2.1,6.9,n).round(1), 0.325),
        'htn'  : cat(['yes','no'], 0.01),
        'dm'   : smear(pd.Series(dm_vals), 0.01),
        'cad'  : cat(['yes','no'], 0.01),
        'appet': cat(['good','poor'], 0.05),
        'pe'   : cat(['yes','no'], 0.01),
        'ane'  : cat(['yes','no'], 0.01),
        'classification': labels,
    })
    print(f"   Shape: {df.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 1: Initial Audit
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*65)
print("STEP 1: INITIAL AUDIT")
print("─"*65)
print(f"  Shape  : {df.shape[0]} rows × {df.shape[1]} columns")
miss = (df.isnull().sum()/len(df)*100).sort_values(ascending=False)
miss = miss[miss > 0]
print(f"  Missing columns: {len(miss)}")
print(miss.round(2).to_string())

# ─────────────────────────────────────────────────────────────
# STEP 2: Drop 'id'  (Issue #5)
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*65)
print("STEP 2: DROP 'id' COLUMN")
print("─"*65)
if 'id' in df.columns:
    df = df.drop(columns=['id'])
    print("  ✅ 'id' dropped")

# ─────────────────────────────────────────────────────────────
# STEP 3: Clean Target Label  (Issue #4)
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*65)
print("STEP 3: CLEAN TARGET LABEL")
print("─"*65)
print(f"  Before: {df['classification'].unique().tolist()[:5]}")
df['classification'] = df['classification'].astype(str).str.strip()
print(f"  After : {df['classification'].unique().tolist()}")
print("  ✅ Trailing tab removed")

# ─────────────────────────────────────────────────────────────
# STEP 4: Clean Dirty Strings  (Issue #3)
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*65)
print("STEP 4: CLEAN DIRTY STRING COLUMNS")
print("─"*65)
cat_cols = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']
for col in cat_cols:
    if col not in df.columns: continue
    dirty = [v for v in df[col].dropna().astype(str).unique() if v != v.strip().lower()]
    cleaned = df[col].astype(str).str.strip().str.lower()
    df[col] = cleaned.replace({'nan': np.nan, 'none': np.nan, '': np.nan})
    if dirty:
        print(f"  ✅ '{col}': cleaned → {dirty[:4]}")
    else:
        print(f"     '{col}': already clean")

# ─────────────────────────────────────────────────────────────
# STEP 5: Fix Wrong Dtypes  (Issue #2)
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*65)
print("STEP 5: FIX WRONG DTYPES  (pcv, wc, rc)")
print("─"*65)
for col in ['pcv','wc','rc']:
    if col not in df.columns: continue
    before = df[col].dtype
    cleaned = df[col].astype(str).str.strip().replace({'nan':np.nan,'none':np.nan,'':np.nan})
    df[col] = pd.to_numeric(cleaned, errors='coerce')
    print(f"  ✅ '{col}': {before} → {df[col].dtype}  (NaN: {df[col].isnull().sum()})")

# ─────────────────────────────────────────────────────────────
# STEP 6: Convert Remaining Object Numerics
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*65)
print("STEP 6: CONVERT REMAINING NUMERIC COLUMNS")
print("─"*65)
skip = set(cat_cols + ['classification','pcv','wc','rc'])
for col in df.columns:
    if col in skip: continue
    if df[col].dtype == object:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"  ✅ '{col}' → {df[col].dtype}  (NaN: {df[col].isnull().sum()})")

# ─────────────────────────────────────────────────────────────
# STEP 7: Missing Value Imputation  (Issue #1)
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*65)
print("STEP 7: MISSING VALUE IMPUTATION")
print("─"*65)
print("  Numeric → Median:")
for col in df.select_dtypes(include=[np.number]).columns:
    n_miss = df[col].isnull().sum()
    if n_miss > 0:
        med = df[col].median()
        df[col] = df[col].fillna(med)
        print(f"   ✅ '{col}': {n_miss} filled → median={med:.2f}")

print("  Categorical → Mode:")
for col in cat_cols:
    if col not in df.columns: continue
    n_miss = df[col].isnull().sum()
    if n_miss > 0:
        mode = df[col].mode()
        if len(mode):
            df[col] = df[col].fillna(mode[0])
            print(f"   ✅ '{col}': {n_miss} filled → mode='{mode[0]}'")

total = df.isnull().sum().sum()
print(f"\n  Remaining missing: {total}")
print("  ✅ All resolved!" if total == 0 else f"  ⚠️ {total} still missing")

# ─────────────────────────────────────────────────────────────
# STEP 8: Label Encoding  (Issue #6)
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*65)
print("STEP 8: LABEL ENCODING")
print("─"*65)
df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})
print(f"  ✅ Target: ckd→1, notckd→0  {df['classification'].value_counts().to_dict()}")

le = LabelEncoder()
for col in cat_cols:
    if col not in df.columns: continue
    vals = sorted(df[col].dropna().unique())
    df[col] = le.fit_transform(df[col].astype(str))
    mapping = {str(v): int(le.transform([str(v)])[0]) for v in vals}
    print(f"  ✅ '{col}': {mapping}")

# ─────────────────────────────────────────────────────────────
# STEP 9: Save Cleaned CSV
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*65)
print("STEP 9: SAVE CLEANED CSV")
print("─"*65)
out_dir  = "clean_datasets"
out_path = f"{out_dir}/kidney_clean.csv"
os.makedirs(out_dir, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"  ✅ Saved → {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)")
print(f"  Shape  : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Missing: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────────
# STEP 10: Train / Test Split + Feature Scaling
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*65)
print("STEP 10: TRAIN/TEST SPLIT  (80% train | 20% test)")
print("─"*65)
X = df.drop(columns=['classification'])
y = df['classification']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train : {X_train.shape[0]} samples")
print(f"  Test  : {X_test.shape[0]} samples")
print(f"  Class balance → CKD={y.sum()} ({y.mean()*100:.1f}%)  Not-CKD={(y==0).sum()}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("  ✅ StandardScaler applied (mean=0, std=1)")

# ─────────────────────────────────────────────────────────────
# STEP 11: Train 6 Models
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*65)
print("STEP 11: TRAINING 6 MODELS...")
print("─"*65)

# (model, needs_scaling)
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
    "Random Forest"       : (RandomForestClassifier(n_estimators=100, random_state=42), False),
    "Decision Tree"       : (DecisionTreeClassifier(random_state=42), False),
    "KNN"                 : (KNeighborsClassifier(n_neighbors=5), True),
    "SVM"                 : (SVC(kernel='rbf', random_state=42), True),
    "XGBoost"             : (XGBClassifier(n_estimators=100, random_state=42,
                                           eval_metric='logloss'), False),
}

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, (model, scaled) in models.items():
    Xtr = X_train_sc if scaled else X_train
    Xte = X_test_sc  if scaled else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cv_m = cross_val_score(model, Xtr, y_train, cv=cv, scoring='accuracy').mean()

    results.append({'Model': name, 'Accuracy': acc, 'Precision': prec,
                    'Recall': rec, 'F1': f1, 'CV': cv_m,
                    '_model': model, '_scaled': scaled})

    bar = "█" * int(acc*20) + "░" * (20-int(acc*20))
    print(f"\n  {name}")
    print(f"  Accuracy  [{bar}] {acc*100:.2f}%")
    print(f"  Precision {prec*100:.2f}%  |  Recall {rec*100:.2f}%  |  F1 {f1*100:.2f}%  |  CV {cv_m*100:.2f}%")

# ─────────────────────────────────────────────────────────────
# STEP 12: Accuracy Comparison Table
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*65)
print("STEP 12: ACCURACY COMPARISON TABLE")
print("─"*65)
res_df = (pd.DataFrame(results)
          .drop(columns=['_model','_scaled'])
          .sort_values('Accuracy', ascending=False)
          .reset_index(drop=True))
for col in ['Accuracy','Precision','Recall','F1','CV']:
    res_df[col] = res_df[col].apply(lambda x: f"{x*100:.2f}%")
print(res_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# STEP 13: Best Model — Detailed Report
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*65)
print("STEP 13: BEST MODEL — DETAILED REPORT")
print("─"*65)
best   = max(results, key=lambda r: r['Accuracy'])
bmodel = best['_model']
Xte_b  = X_test_sc if best['_scaled'] else X_test
y_pred_b = bmodel.predict(Xte_b)

print(f"\n  🏆 Best Model : {best['Model']}")
print(f"  Accuracy     : {best['Accuracy']*100:.2f}%")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_b).ravel()
print(f"\n  Confusion Matrix:")
print(f"  ┌─────────────────────────────┐")
print(f"  │          Predicted           │")
print(f"  │        Not-CKD    CKD        │")
print(f"  │ Not-CKD   {tn:5}     {fp:5}      │")
print(f"  │ CKD       {fn:5}     {tp:5}      │")
print(f"  └─────────────────────────────┘")
print(f"\n  TP (CKD caught)        : {tp}")
print(f"  TN (Not-CKD correct)   : {tn}")
print(f"  FP (Wrong CKD alarm)   : {fp}  ← Type I Error")
print(f"  FN (CKD missed!)       : {fn}  ← Type II Error ⚠️")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_b, target_names=['Not-CKD','CKD']))

if hasattr(bmodel, 'feature_importances_'):
    print(f"  Top 10 Features ({best['Model']}):")
    fi = pd.Series(bmodel.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    for feat, imp in fi.items():
        bar = "█" * int(imp*100)
        print(f"   {feat:<10} [{bar:<20}] {imp*100:.2f}%")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
for r in sorted(results, key=lambda x: x['Accuracy'], reverse=True):
    medal = "🥇" if r['Model'] == best['Model'] else "  "
    print(f"  {medal} {r['Model']:<22} Accuracy: {r['Accuracy']*100:.2f}%  F1: {r['F1']*100:.2f}%")
print(f"\n  ✅ Best Model : {best['Model']}  ({best['Accuracy']*100:.2f}%)")
print(f"  ✅ Clean CSV  : {out_path}")
print(f"  ✅ Dataset    : {df.shape[0]} rows, {X.shape[1]} features")
print("=" * 65)
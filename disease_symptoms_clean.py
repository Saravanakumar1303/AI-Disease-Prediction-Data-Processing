import os, time, warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────
INPUT_PATH  = "raw_datasets/DiseaseAndSymptoms.csv"
OUTPUT_PATH = "clean_datasets/DiseaseAndSymptoms_Clean.csv"
os.makedirs("clean_datasets", exist_ok=True)

SEP = "=" * 65

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def row(label, value):
    print(f"  {label:<40}: {value}")

def divider():
    print(f"  {'-' * 60}")


# ══════════════════════════════════════════════════════════════
#  STEP 1 — LOAD RAW DATA
# ══════════════════════════════════════════════════════════════
section("STEP 1 — LOAD RAW DATA")

t0 = time.time()
df_raw = pd.read_csv(INPUT_PATH)
sym_cols = [c for c in df_raw.columns if c.startswith("Symptom")]

row("File", INPUT_PATH)
row("Raw shape", f"{df_raw.shape[0]} rows  ×  {df_raw.shape[1]} cols")
row("Disease col + symptom cols", f"1 + {len(sym_cols)}")
row("Unique diseases", df_raw["Disease"].nunique())
row("Rows per disease (raw)", df_raw["Disease"].value_counts().iloc[0])
row("Load time", f"{time.time()-t0:.3f}s")


# ══════════════════════════════════════════════════════════════
#  STEP 2 — CLEAN & REMOVE DUPLICATES
# ══════════════════════════════════════════════════════════════
section("STEP 2 — CLEAN & REMOVE DUPLICATES")

df = df_raw.copy()

# Normalize
for c in sym_cols:
    df[c] = df[c].str.strip().str.lower()
df["Disease"] = df["Disease"].str.strip()

# Build a frozenset key per row (order-insensitive)
def sym_set(row):
    return frozenset(
        row[c] for c in sym_cols
        if pd.notna(row[c]) and str(row[c]).strip()
    )

df["_key"] = df.apply(sym_set, axis=1)
dup_key    = df["Disease"] + "||" + df["_key"].astype(str)
n_dups     = dup_key.duplicated().sum()

row("Normalization", "strip + lowercase applied")
row("Duplicate detection", "Disease + symptom-set (order-insensitive)")
row("Duplicates found", f"{n_dups}  ({n_dups/len(df)*100:.1f}% of raw data)")

df_clean = df[~dup_key.duplicated()].copy().reset_index(drop=True)

row("Rows before dedup", len(df))
row("Rows after  dedup", len(df_clean))
row("Rows removed", len(df) - len(df_clean))

divider()
print("  Unique rows per disease after dedup:")
for disease, cnt in df_clean["Disease"].value_counts().items():
    bar = "█" * cnt
    print(f"    {disease:<50} {cnt:>3}  {bar}")


# ══════════════════════════════════════════════════════════════
#  STEP 3 — BINARY ENCODING  (disease name as columns)
# ══════════════════════════════════════════════════════════════
section("STEP 3 — BINARY ENCODING")

# All unique symptoms (sorted = stable column order)
all_syms = sorted({
    v
    for c in sym_cols
    for v in df_clean[c].dropna()
    if str(v).strip()
})

row("Unique symptoms (= new columns)", len(all_syms))
row("Encoding", "1 if symptom present in that row, else 0")

t0 = time.time()
bin_rows = []
for _, r in df_clean.iterrows():
    present = r["_key"]
    bin_rows.append([1 if s in present else 0 for s in all_syms])

X_arr = np.array(bin_rows, dtype=np.int8)
y_arr = df_clean["Disease"].values

# Final ML dataframe:  Disease | sym_1 | sym_2 | ... | sym_131
df_ml = pd.DataFrame(X_arr, columns=all_syms)
df_ml.insert(0, "Disease", y_arr)

syms_per = X_arr.sum(axis=1)
row("Encoding time", f"{time.time()-t0:.3f}s")
row("Final shape", f"{df_ml.shape[0]} rows  ×  {df_ml.shape[1]} cols")
row("Column[0]", "Disease  (label)")
row(f"Columns[1..{len(all_syms)}]", "binary symptom flags  (0 / 1)")
row("Avg symptoms per row", f"{syms_per.mean():.2f}")
row("Max symptoms per row", int(syms_per.max()))
row("Min symptoms per row", int(syms_per.min()))
row("Matrix sparsity", f"{(1-X_arr.mean())*100:.1f}% zeros")

divider()
print("  All 131 binary symptom columns:")
for i, s in enumerate(all_syms, 1):
    print(f"    {i:>3}. {s}")


# ══════════════════════════════════════════════════════════════
#  STEP 4 — SAVE CLEAN CSV
# ══════════════════════════════════════════════════════════════
section("STEP 4 — SAVE CLEAN CSV")

df_ml.to_csv(OUTPUT_PATH, index=False)
kb = os.path.getsize(OUTPUT_PATH) / 1024

row("Saved to", OUTPUT_PATH)
row("Shape", f"{df_ml.shape[0]} rows  ×  {df_ml.shape[1]} cols")
row("File size", f"{kb:.1f} KB")

divider()
print("  Preview — first 5 rows, first 9 cols:\n")
print(df_ml.iloc[:5, :9].to_string(index=False))


# ══════════════════════════════════════════════════════════════
#  STEP 5 — ML TRAINING  (Random Forest)
# ══════════════════════════════════════════════════════════════
section("STEP 5 — ML TRAINING")

X = df_ml[all_syms].values
y = df_ml["Disease"].values

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

row("Algorithm", "Random Forest")
row("n_estimators", 200)
row("max_depth", "None  (fully grown)")
row("random_state", 42)
row("Train rows", X_tr.shape[0])
row("Test  rows", X_te.shape[0])
row("Features", X_tr.shape[1])

t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=200, max_depth=None,
    random_state=42,  n_jobs=-1
)
rf.fit(X_tr, y_tr)
t_train = time.time() - t0

row("Training time", f"{t_train:.3f}s")
row("Train accuracy", f"{rf.score(X_tr, y_tr)*100:.2f}%")

divider()
print("  Per-class train/test split:")
tr_c = pd.Series(y_tr).value_counts().sort_index()
te_c = pd.Series(y_te).value_counts().sort_index()
for d in sorted(np.unique(y)):
    print(f"    {d:<50}  train={tr_c.get(d,0)}  test={te_c.get(d,0)}")


# ══════════════════════════════════════════════════════════════
#  STEP 6 — TRAINING REPORT
# ══════════════════════════════════════════════════════════════
section("STEP 6 — TRAINING REPORT")

y_pred   = rf.predict(X_te)
acc      = accuracy_score(y_te, y_pred)
prec     = precision_score(y_te, y_pred, average="macro",    zero_division=0)
rec      = recall_score(y_te, y_pred,    average="macro",    zero_division=0)
f1_mac   = f1_score(y_te, y_pred,        average="macro",    zero_division=0)
f1_wt    = f1_score(y_te, y_pred,        average="weighted", zero_division=0)

print("  ── Test Set ──────────────────────────────────────────")
row("Test Accuracy",         f"{acc*100:.4f}%")
row("Precision  (macro)",    f"{prec*100:.4f}%")
row("Recall     (macro)",    f"{rec*100:.4f}%")
row("F1-Score   (macro)",    f"{f1_mac*100:.4f}%")
row("F1-Score   (weighted)", f"{f1_wt*100:.4f}%")
row("Misclassifications",    f"{(y_te != y_pred).sum()} / {len(y_te)}")

divider()
print("  ── 5-Fold Stratified Cross-Validation ───────────────")
skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(rf, X, y, cv=skf, scoring="accuracy")
cv_f1  = cross_val_score(rf, X, y, cv=skf, scoring="f1_macro")
for i, (a, f) in enumerate(zip(cv_acc, cv_f1), 1):
    row(f"  Fold {i}", f"Accuracy={a*100:.4f}%   F1={f*100:.4f}%")
divider()
row("CV Accuracy   mean ± std", f"{cv_acc.mean()*100:.4f}%  ±  {cv_acc.std()*100:.4f}%")
row("CV F1-macro   mean ± std", f"{cv_f1.mean()*100:.4f}%  ±  {cv_f1.std()*100:.4f}%")

divider()
print("  ── Per-Class Report (test set) ───────────────────────\n")
print(classification_report(y_te, y_pred, zero_division=0))

divider()
print("  ── Feature Importance  (top 20) ─────────────────────")
imp = pd.Series(rf.feature_importances_, index=all_syms).sort_values(ascending=False)
for rank, (sym, val) in enumerate(imp.head(20).items(), 1):
    bar = "▓" * int(val * 800)
    print(f"    {rank:>2}. {sym:<38} {val:.4f}  {bar}")


# ══════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
section("FINAL SUMMARY")

print(f"""
  ┌─ RAW DATA ────────────────────────────────────────┐
    Rows          : 4920
    Columns       : 18  (Disease + Symptom_1..17)
    Duplicates    : {n_dups} rows  ({n_dups/len(df)*100:.1f}%)
  └────────────────────────────────────────────────────┘

  ┌─ CLEAN DATA ──────────────────────────────────────┐
    Rows          : {df_ml.shape[0]}
    Columns       : {df_ml.shape[1]}  (Disease + 131 binary flags)
    Format        : 0 / 1  per symptom
    Duplicates    : 0
    Saved to      : {OUTPUT_PATH}
  └────────────────────────────────────────────────────┘

  ┌─ ML RESULTS ──────────────────────────────────────┐
    Algorithm     : Random Forest  (200 trees)
    Train / Test  : {X_tr.shape[0]} / {X_te.shape[0]}  rows
    Test Accuracy : {acc*100:.2f}%
    F1 (macro)    : {f1_mac*100:.2f}%
    CV Accuracy   : {cv_acc.mean()*100:.2f}% ± {cv_acc.std()*100:.2f}%
  └────────────────────────────────────────────────────┘

  ✅  Done.  Output → {OUTPUT_PATH}
""")
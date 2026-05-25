import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df =pd.read_csv("raw_datasets/diabetes.csv")

# 1. These columns can't be 0 medically
zero_not_allowed = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Step_1: Replace Zero with Nan
df[zero_not_allowed] = df[zero_not_allowed].replace(0, np.nan)

# Step_2: Replace Nan with Median(Median is better than Mean for Skewed data)
df[zero_not_allowed] = df[zero_not_allowed].fillna(df[zero_not_allowed].median())

print("Missing Values after fix:\n",df.isnull().sum())

# 2. Cap Outliers using IQR

def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df[column] = df[column].clip(lower = Q1-1.5*IQR, upper = Q3+1.5*IQR)
    return df

for col in ['Insulin', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction']:
    df = cap_outliers(df, col)

#---Split---
X = df.drop('Outcome', axis=1)   # ← இந்த line இருக்கா?
y = df['Outcome']              

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ── Scale ─────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Train ─────────────────────────────────────────────
model = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy*100:.2f}%")
print(classification_report(y_test, y_pred))
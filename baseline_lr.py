# notebooks/baseline_lr.py
# A single-script baseline that:
# 1) downloads UCI Cleveland data, 
# 2) applies cohort rules, 
# 3) writes data/interim/heart_cleveland_interim.csv,
# 4) trains a LogisticRegression baseline and prints AUROC and accuracy,
# 5) produces subgroup table and saves to reports/subgroup_table.csv

import os
import io
import urllib.request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# --- directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/interim", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("dashboard", exist_ok=True)

# --- download the dataset (processed Cleveland)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
raw_path = "data/raw/processed.cleveland.data"
if not os.path.exists(raw_path):
    print("Downloading dataset from UCI...")
    urllib.request.urlretrieve(url, raw_path)
    print("Downloaded to", raw_path)
else:
    print("Using cached file at", raw_path)

# --- columns as per UCI (14 attributes commonly used)
cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]

df = pd.read_csv(raw_path, header=None, names=cols, na_values=['?'])
print("Raw shape:", df.shape)
print(df.head())

# --- Cohort rules (from configs/cohort.yml)
# Inclusion: age >= 18, drop rows with missing outcome, drop rows with any missing model feature
df = df[df["age"] >= 18]
df = df.dropna(subset=["num"])
df = df.dropna()  # drop any rows with missing features (simple rule for Week1)

# Binarize outcome: 0 -> 0; 1-4 -> 1
df["y"] = (df["num"] > 0).astype(int)

# Save interim
interim_path = "data/interim/heart_cleveland_interim.csv"
df.to_csv(interim_path, index=False)
print("Saved interim cohort to", interim_path, "shape:", df.shape)

# --- Basic descriptive stats
print("\nOutcome distribution:")
print(df["y"].value_counts(normalize=False))
print("\nAge, sex distribution:")
print(df.groupby("sex")["y"].agg(["count","mean"]))

# --- Prepare features for baseline model
feature_cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
X = df[feature_cols].copy()
y = df["y"].copy()

# Simple encoding: many fields are numeric already; ensure dtype numeric
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors='coerce')
X = X.fillna(0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_s, y_train)

y_pred_proba = clf.predict_proba(X_test_s)[:,1]
y_pred = clf.predict(X_test_s)

auc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)

print(f"\nBaseline Logistic Regression results — AUROC: {auc:.4f}, Accuracy: {acc:.4f}")
print("\nClassification report (test):")
print(classification_report(y_test, y_pred))

# --- subgroup analysis (Pair C)
def age_band(age):
    if age < 40: return "<40"
    elif age < 60: return "40-59"
    else: return "60+"

df['age_band'] = df['age'].apply(age_band)
subgroup = df.groupby(['age_band','sex']).agg(
    n=('y','count'),
    outcome_rate=('y','mean')
).reset_index()
subgroup['outcome_rate'] = subgroup['outcome_rate'].round(3)
subgroup.to_csv("reports/subgroup_table.csv", index=False)
print("\nSaved subgroup table to reports/subgroup_table.csv")
print(subgroup)

# Save model (optional)
import joblib
joblib.dump({"model":clf, "scaler":scaler, "features":feature_cols}, "reports/baseline_lr.joblib")
print("Saved baseline model to reports/baseline_lr.joblib")

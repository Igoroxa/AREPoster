import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt

# ---------------------------------------
# STEP 1: Load and clean the dataset
# ---------------------------------------
print("🔍 Loading data...")
# revert to original CSV
df = pd.read_csv("earthquakes_subduction_expanded.csv", low_memory=False)
label_column = "subduction_flag"

# Pick only the columns you know exist in this file:
numeric_cols = [
    "latitude", "longitude", "depth", "mag",
    "nst", "gap", "dmin", "rms",
    "year", "month", "day_of_year",
]
categorical_cols = ["magType"]

# drop rows missing any of those + the label
df = df.dropna(subset=numeric_cols + categorical_cols + [label_column])

# ---------------------------------------
# STEP 2: One-hot encode categoricals
# ---------------------------------------
print("🎛️  Encoding categorical features...")
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = ohe.fit_transform(df[categorical_cols])
X_cat_df = pd.DataFrame(
    X_cat,
    columns=ohe.get_feature_names_out(categorical_cols),
    index=df.index
)

# Combine numeric + one-hot
X = pd.concat([df[numeric_cols], X_cat_df], axis=1)
y = df[label_column].astype(int)

# ---------------------------------------
# STEP 3: Train-test split & scale
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---------------------------------------
# STEP 4: Train XGBoost model
# ---------------------------------------
print("🚀 Training XGBoost model...")
pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
base_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    scale_pos_weight=pos_weight,
    random_state=42,
)

param_distributions = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.5, 1, 2],
    "min_child_weight": [1, 3, 5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    base_model,
    param_distributions,
    n_iter=10,
    scoring="average_precision",
    n_jobs=-1,
    cv=cv,
    verbose=1,
    random_state=42,
)

search.fit(X_train_scaled, y_train)
model = search.best_estimator_

# ---------------------------------------
# STEP 5: Evaluate model
# ---------------------------------------
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
pos_probs = y_pred_prob[y_test == 1]
threshold = pos_probs.min() if len(pos_probs) > 0 else 0.5
y_pred      = (y_pred_prob >= threshold).astype(int)
print(f"Using threshold {threshold:.4f} for 100% recall")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"🌐 ROC AUC: {roc_auc:.3f}")

# ---------------------------------------
# STEP 6: Plots
# ---------------------------------------

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
ap = average_precision_score(y_test, y_pred_prob)
plt.figure()
plt.plot(recall, precision, label=f"AP = {ap:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.show()

# Feature importance
plt.figure(figsize=(8,6))
xgb.plot_importance(model, max_num_features=10, height=0.5)
plt.title("Top 10 Feature Importances")
plt.show()

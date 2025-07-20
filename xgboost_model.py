import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
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
    "latitude","longitude","depth","mag",
    "year","month","day_of_year"
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
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
    max_depth=6,
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ---------------------------------------
# STEP 5: Evaluate model
# ---------------------------------------
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred      = (y_pred_prob >= 0.5).astype(int)

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

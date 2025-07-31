#!/usr/bin/env python3
"""
train_xgboost_subduction_recallOptimized.py

This version aggressively reduces false negatives by:
 - Searching a wider `scale_pos_weight` range
 - Optimizing hyperparameters for **recall** instead of F1
 - Automatically tuning threshold to maximize recall at a user-defined precision floor
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import joblib
import argparse


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    numeric_features = ['latitude', 'longitude', 'depth', 'mag', 'year', 'month', 'day_of_year']
    categorical_features = ['magType']
    label = 'subduction_flag'
    df = df.dropna(subset=numeric_features + categorical_features + [label])
    X = df[numeric_features + categorical_features]
    y = df[label]
    return X, y


def build_pipeline():
    categorical_features = ['magType']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ],
        remainder='passthrough'
    )

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'gpu_id': 0,
        'random_state': 42,
        'verbosity': 1
    }

    clf = XGBClassifier(**xgb_params)
    return Pipeline([('preprocess', preprocessor), ('clf', clf)])


def tune_model(pipeline, X_train, y_train, X_valid, y_valid):
    # Wider search for scale_pos_weight to boost recall
    base_w = np.sum(y_train == 0) / np.sum(y_train == 1)
    weight_options = [base_w * f for f in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]]

    param_dist = {
        'clf__n_estimators': [400, 600, 800, 1000],
        'clf__max_depth': [3, 5, 7, 9],
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__subsample': [0.7, 0.85, 1.0],
        'clf__colsample_bytree': [0.7, 0.85, 1.0],
        'clf__gamma': [0, 1, 5],
        'clf__min_child_weight': [1, 3, 5],
        'clf__scale_pos_weight': weight_options
    }

    # Optimize for recall
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=50,
        scoring='recall',
        cv=4,
        random_state=42,
        n_jobs=1
    )
    search.fit(X_train, y_train)
    return search


def find_best_threshold(y_true, y_prob, min_precision=0.8):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # Only consider thresholds where precision >= min_precision
    mask = precision >= min_precision
    if not mask.any():
        best = 0.5
        print(f"No threshold meets precision >= {min_precision:.2f}, using 0.5")
    else:
        rec = recall[mask]
        best = thresholds[mask][np.argmax(rec[:-1])]
        print(f"Threshold for max recall at precision >= {min_precision:.2f}: {best:.3f}")
    return best


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost optimizing for recall to reduce false negatives.")
    parser.add_argument('--data', type=str, default='earthquakes_subduction_expanded.csv')
    parser.add_argument('--test-size', type=float, default=0.3)
    parser.add_argument('--min-prec', type=float, default=0.8,
                        help="Minimum precision constraint when tuning threshold for recall.")
    args = parser.parse_args()

    X, y = load_data(args.data)
    X_tr, X_temp, y_tr, y_temp = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    pipeline = build_pipeline()
    print("Tuning hyperparameters to maximize recall...")
    search = tune_model(pipeline, X_tr, y_tr, X_val, y_val)

    print("\nBest params (recall):", search.best_params_)
    print(f"Best CV recall: {search.best_score_:.4f}\n")

    best_model = search.best_estimator_
    y_proba = best_model.predict_proba(X_test)[:, 1]
    thresh = find_best_threshold(y_test, y_proba, min_precision=args.min_prec)
    y_pred = (y_proba >= thresh).astype(int)

    print(f"\nClassification Report (threshold={thresh:.3f}):")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(best_model, 'xgb_subduction_model_recallOptimized.joblib')
    print("\nOptimized model saved to xgb_subduction_model_recallOptimized.joblib")

if __name__ == '__main__':
    main()

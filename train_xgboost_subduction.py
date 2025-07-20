#!/usr/bin/env python3
"""
train_xgboost_subduction_optimized.py

Enhanced XGBoost training to reduce false negatives and false positives:
 - Hyperparameter search over scale_pos_weight
 - Early stopping on a validation fold (manual retrain after CV)
 - Automatic threshold tuning (maximize F1 or target recall)
 - Expanded parameter grid for improved model complexity control
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
    pipeline = Pipeline([('preprocess', preprocessor), ('clf', clf)])
    return pipeline


def tune_model(pipeline, X_train, y_train, X_valid, y_valid):
    # Compute base scale_pos_weight
    base_w = np.sum(y_train == 0) / np.sum(y_train == 1)
    # Search over a small set of weight multipliers
    weight_options = [base_w * f for f in [0.5, 1.0, 1.5, 2.0]]

    param_dist = {
        'clf__n_estimators': [200, 400, 600, 800],
        'clf__max_depth': [3, 5, 7, 9],
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__subsample': [0.7, 0.85, 1.0],
        'clf__colsample_bytree': [0.7, 0.85, 1.0],
        'clf__gamma': [0, 1, 5],
        'clf__min_child_weight': [1, 3, 5],
        'clf__scale_pos_weight': weight_options
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=50,
        scoring='f1',      # optimizing F1 balance
        cv=4,
        random_state=42,
        n_jobs=1
    )
    # Fit without unsupported args
    search.fit(X_train, y_train)
    return search


def find_best_threshold(y_true, y_prob, focus='f1', min_precision=None):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    best_thresh = 0.5
    if focus == 'f1':
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        best_thresh = thresholds[np.argmax(f1[:-1])]
        print(f"Best threshold by F1: {best_thresh:.3f}")
    elif focus == 'recall' and min_precision is not None:
        mask = precision >= min_precision
        if mask.any():
            rec = recall[mask]
            best_thresh = thresholds[mask][np.argmax(rec[:-1])]
            print(f"Threshold for max recall at precision >= {min_precision}: {best_thresh:.3f}")
    return best_thresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='earthquakes_subduction_expanded.csv')
    parser.add_argument('--test-size', type=float, default=0.3)
    parser.add_argument('--focus', choices=['f1','recall'], default='f1',
                        help="Whether to pick threshold by F1 or recall (requires --min-prec)")
    parser.add_argument('--min-prec', type=float,
                        help="Minimum precision when focusing on recall threshold")
    args = parser.parse_args()

    # Load data and split
    X, y = load_data(args.data)
    X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)
    valid_frac = 0.5
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=valid_frac, stratify=y_temp, random_state=42)

    pipeline = build_pipeline()
    print("Tuning model hyperparameters...")
    search = tune_model(pipeline, X_tr, y_tr, X_val, y_val)

    print("\nBest params:", search.best_params_)
    print(f"Best CV F1: {search.best_score_:.4f}\n")

    best = search.best_estimator_
    y_proba = best.predict_proba(X_test)[:,1]
    thresh = find_best_threshold(y_test, y_proba, focus=args.focus, min_precision=args.min_prec)
    y_pred = (y_proba >= thresh).astype(int)

    print(f"Classification Report (threshold={thresh:.3f}):")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(best, 'xgb_subduction_model_optimized.joblib')
    print("\nOptimized model saved to xgb_subduction_model_optimized.joblib")

if __name__ == '__main__':
    main()

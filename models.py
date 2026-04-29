"""
models.py - Core ML Module for ASD Risk Prediction
====================================================
Trains multiple classifiers on train.csv, compares performance,
displays feature importance, and saves the best model as best_model.pkl.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# -----------------------------------------------
# 1. DATA LOADING
# -----------------------------------------------
def load_data(path: str = "train.csv") -> pd.DataFrame:
    """Load the ASD dataset from a CSV file."""
    df = pd.read_csv(path)
    print(f"[OK] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# -----------------------------------------------
# 2. DATA PREPROCESSING
# -----------------------------------------------
def preprocess(df: pd.DataFrame):
    """
    Clean the data, encode categoricals, and return
    (X, y, encoders_dict) ready for modelling.
    """
    # Drop leakage / irrelevant columns
    drop_cols = ["ID", "age_desc", "result"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Target column
    target = "Class/ASD"

    # -- Handle missing / noisy values --
    df["ethnicity"] = df["ethnicity"].replace({"?": "Others", "others": "Others"})
    df["relation"] = df["relation"].replace(
        {"?": "Others", "Relative": "Others", "Parent": "Others",
         "Health care professional": "Others"}
    )

    # Fill any remaining NaN with column mode (categorical) or median (numeric)
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == "object":
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

    # -- Encode categorical columns --
    encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Save encoders for backend integration
    with open("encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    print("[OK] Encoders saved -> encoders.pkl")

    X = df.drop(columns=[target])
    y = df[target]

    print(f"[OK] Preprocessing complete - Features: {X.shape[1]}, Target classes: {y.nunique()}")
    return X, y, encoders


# -----------------------------------------------
# 3. TRAIN-TEST SPLIT
# -----------------------------------------------
def split_data(X, y, test_size=0.2, random_state=42):
    """80/20 stratified split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[OK] Split - Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# -----------------------------------------------
# 4. MODEL TRAINING
# -----------------------------------------------
def get_models():
    """Return a dict of model name -> model instance."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    }


def train_models(models: dict, X_train, y_train):
    """Fit every model and return the same dict (now trained)."""
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"  [+] {name} trained")
    return models


# -----------------------------------------------
# 5. EVALUATION METRICS
# -----------------------------------------------
def evaluate_models(models: dict, X_test, y_test):
    """
    For each model compute accuracy, precision, recall, and confusion matrix.
    Returns a list of result dicts.
    """
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        results.append({
            "Model":     name,
            "Accuracy":  acc,
            "Precision": prec,
            "Recall":    rec,
            "Confusion": cm,
        })
    return results


# -----------------------------------------------
# 6. MODEL COMPARISON
# -----------------------------------------------
def print_comparison(results: list):
    """Print a neat table and highlight the best model."""
    print("\n" + "=" * 65)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['Model']:<25} {r['Accuracy']:>10.4f} {r['Precision']:>10.4f} {r['Recall']:>10.4f}")
    print("=" * 65)

    # Confusion matrices
    for r in results:
        print(f"\n[*] Confusion Matrix - {r['Model']}:")
        print(r["Confusion"])

    best = max(results, key=lambda x: x["Accuracy"])
    print(f"\n>>> BEST MODEL: {best['Model']} performed best with {best['Accuracy']*100:.2f}% accuracy\n")
    return best["Model"]


# -----------------------------------------------
# 7. PROBABILITY-BASED RISK LEVELS
# -----------------------------------------------
def risk_levels(model, X_test, y_test):
    """
    Use predict_proba to assign risk levels.
      0-30 %  -> Low Risk
     30-70 %  -> Medium Risk
     70-100%  -> High Risk
    """
    probas = model.predict_proba(X_test)[:, 1]  # probability of class = 1 (ASD)

    levels = []
    for p in probas:
        pct = p * 100
        if pct <= 30:
            levels.append("[GREEN] Low Risk")
        elif pct <= 70:
            levels.append("[YELLOW] Medium Risk")
        else:
            levels.append("[RED] High Risk")

    # Print a sample of 10 predictions
    print("\n-- Sample Risk Predictions (first 10 test rows) --")
    print(f"{'Actual':<10} {'Prob(%)':<10} {'Risk Level'}")
    print("-" * 40)
    for actual, prob, level in zip(y_test.values[:10], probas[:10], levels[:10]):
        print(f"{actual:<10} {prob*100:<10.2f} {level}")


# -----------------------------------------------
# 8. FEATURE IMPORTANCE
# -----------------------------------------------
def show_feature_importance(models: dict, feature_names: list):
    """Display feature importance for tree-based models."""
    for name in ["Decision Tree", "Random Forest"]:
        model = models[name]
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print(f"\n-- Feature Importance: {name} --")
        print(f"{'Rank':<6} {'Feature':<25} {'Importance':<12}")
        print("-" * 45)
        for rank, idx in enumerate(indices, 1):
            print(f"{rank:<6} {feature_names[idx]:<25} {importances[idx]:<12.4f}")


# -----------------------------------------------
# 9. SAVE BEST MODEL
# -----------------------------------------------
def save_best_model(models: dict, best_name: str, path="best_model.pkl"):
    """Pickle the best model to disk."""
    with open(path, "wb") as f:
        pickle.dump(models[best_name], f)
    print(f"[OK] Best model ({best_name}) saved -> {path}")


# -----------------------------------------------
# 10. MAIN PIPELINE
# -----------------------------------------------
def main():
    print("=" * 65)
    print("   ASD Risk Prediction - ML Training Pipeline")
    print("=" * 65)

    # Step 1 - Load
    df = load_data("train.csv")

    # Step 2 - Preprocess
    X, y, encoders = preprocess(df)

    # Step 3 - Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4 - Train
    print("\n[*] Training models...")
    models = get_models()
    train_models(models, X_train, y_train)

    # Step 5 - Evaluate
    print("\n[*] Evaluating models...")
    results = evaluate_models(models, X_test, y_test)

    # Step 6 - Compare
    best_name = print_comparison(results)

    # Step 7 - Risk levels (using best model)
    risk_levels(models[best_name], X_test, y_test)

    # Step 8 - Feature importance
    show_feature_importance(models, X.columns.tolist())

    # Step 9 - Save
    save_best_model(models, best_name)

    print("\n[OK] Pipeline complete!")


if __name__ == "__main__":
    main()

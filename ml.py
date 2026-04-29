import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
import pickle

# read the csv data to a pandas dataframein.csv
df = pd.read_csv(r"C:\Users\LENOVO\Downloads\asd final\train.csv")

# ---------------- PHASE 1: BASIC CLEANING ----------------

# drop leakage & irrelevant columns

df = df.drop(columns=["ID", "age_desc", "result"])

# handle missing values in ethnicity & relation
df["ethnicity"] = df["ethnicity"].replace({"?": "Others", "others": "Others"})
df["relation"] = df["relation"].replace(
    {"?": "Others", "Relative": "Others", "Parent": "Others", "Health care professional": "Others"}
)

# ---------------- PHASE 0: MAKING PLOTS ----------------

# set the desired theme
sns.set_theme(style="darkgrid")

# Histogram for "age"

sns.histplot(df["age"], kde=True)
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


 # countplot for target column (Class/ASD)
sns.countplot(x=df["Class/ASD"])
plt.title("Count Plot for Class/ASD")
plt.xlabel("Class/ASD")
plt.ylabel("Count")
plt.show()


# ---------------- PHASE 2: ENCODING ----------------

# identify categorical columns
object_columns = df.select_dtypes(include=["object"]).columns
print("Categorical columns:", object_columns)

encoders = {}

for column in object_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# save encoders
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("Encoders saved!")

# Correlation Matrix

plt.figure(figsize=(14,10))

correlation_matrix = df.corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Feature Correlation Matrix", fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.show()


# ---------------- PHASE 3: DATA SPLIT ----------------

X = df.drop(columns=["Class/ASD"])
y = df["Class/ASD"]

# 1. First split: Train+Val (80%) and Test (20%)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Second split: Train (80% of 80% = 64%) and Validation (20% of 80% = 16%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f"Dataset Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# ---------------- PHASE 4: SMOTE ----------------

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("After SMOTE:", y_train_smote.value_counts())

# ---------------- PHASE 5: MODEL TRAINING ----------------

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss")
}

cv_scores = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    if model_name == "XGBoost":
        # Using Validation set for XGBoost early stopping / monitoring
        model.fit(X_train_smote, y_train_smote, eval_set=[(X_train_smote, y_train_smote), (X_val, y_val)], verbose=False)
        results = model.evals_result()
        
        # Plotting Loss and Accuracy for XGBoost (Validation Phase)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(results['validation_0']['logloss'], label='Train Loss')
        plt.plot(results['validation_1']['logloss'], label='Val Loss')
        plt.title('XGBoost Log Loss')
        plt.xlabel('Epochs')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        # Note: XGBoost doesn't return accuracy by default in evals_result unless specified, 
        # but logloss is the standard proxy for training progress.
        plt.plot(np.exp(-np.array(results['validation_0']['logloss'])), label='Train Accuracy (Proxy)')
        plt.plot(np.exp(-np.array(results['validation_1']['logloss'])), label='Val Accuracy (Proxy)')
        plt.title('Training Progress (Accuracy Proxy)')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig("training_validation_curves.png")
        plt.show()

    scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring="accuracy")
    cv_scores[model_name] = np.mean(scores)
    print(f"{model_name} Cross-Validation Accuracy: {np.mean(scores):.4f}")
    print("-"*50)

print("CV Summary:", cv_scores)

# ---------------- PHASE 6: HYPERPARAMETER TUNING  ----------------



epochs = 20
print(f"Running hyperparameter tuning for {epochs} epochs...")

param_grid_rf = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

rf = RandomForestClassifier(random_state=42)

random_search_rf = RandomizedSearchCV(
    rf,
    param_distributions=param_grid_rf,
    n_iter=epochs,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

random_search_rf.fit(X_train_smote, y_train_smote)
print(f"{epochs} epochs of training completed.")

best_model = random_search_rf.best_estimator_
print("Best RF Params:", random_search_rf.best_params_)
print("Best CV Accuracy:", random_search_rf.best_score_)

# Validate on Validation Set
y_val_pred = best_model.predict(X_val)
print("\n--- VALIDATION RESULTS ---")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred))

# Test on Final Test Set
y_test_pred = best_model.predict(X_test)
print("\n--- FINAL TEST RESULTS ---")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))

# Plot ROC and Precision-Recall Curves
y_score = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
precision, recall, _ = precision_recall_curve(y_test, y_score)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('Receiver Operating Characteristic (ROC)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {average_precision_score(y_test, y_score):.2f})')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig("evaluation_curves.png")
plt.show()

# ---------------- PHASE 8: SAVE FINAL MODEL ----------------

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("✅ best_model.pkl saved successfully!")

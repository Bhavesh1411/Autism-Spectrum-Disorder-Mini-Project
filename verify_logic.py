import pandas as pd
import pickle
import numpy as np
import os

# Load model and encoders
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Load data
df = pd.read_csv("train.csv")

# Same cleaning as in ml.py
df.replace("?", np.nan, inplace=True)
df["ethnicity"] = df["ethnicity"].fillna("Others")
df["relation"] = df["relation"].fillna("Others")
df.dropna(inplace=True)

# Re-identify test indices after dropna
pos_idx = df[df['Class/ASD'] == 1].index[0]
neg_idx = df[df['Class/ASD'] == 0].index[0]
test_indices = [pos_idx, neg_idx]

for idx in test_indices:
    row = df.iloc[idx].copy()
    actual = row['Class/ASD']
    
    # Preprocess row (remove target and unused cols)
    input_data = row.drop(['ID', 'age_desc', 'result', 'Class/ASD']).to_frame().T
    
    # Encode
    for col in input_data.columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col].astype(str))
            
    # Align features
    input_data = input_data[model.feature_names_in_]
    
    # Predict
    proba = model.predict_proba(input_data)[0]
    pred = int(np.argmax(proba))
    
    print(f"Row {idx}: Actual={actual}, Predicted={pred}, Proba={proba}")

print("\nVerification Script Completed.")

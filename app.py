import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model and encoders
with open(r"C:\Users\LENOVO\Downloads\asd final\best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(r"C:\Users\LENOVO\Downloads\asd final\encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.set_page_config(page_title="ASD Screening App", page_icon="🧠", layout="centered")

st.title("🧠 Autism Spectrum Disorder (ASD) Screening")
st.write("Answer the questions below to get ASD screening result (Model-based)")

# ---------------- User Inputs ----------------

questions = [
    "1. Difficulty making eye contact?",
    "2. Trouble understanding others' feelings?",
    "3. Prefers being alone rather than with others?",
    "4. Uncomfortable in social situations?",
    "5. Difficulty starting/continuing conversations?",
    "6. Takes things very literally?",
    "7. Gets overly focused on one topic/interest?",
    "8. Upset by small changes in routine?",
    "9. Difficulty understanding social rules?",
    "10. Struggles to express emotions clearly?"
]

opts = ["Yes", "No"]
A_vals = []

st.subheader("📝 Screening Questions")
for q in questions:
    ans = st.selectbox(q, opts, index=1)
    A_vals.append(1 if ans == "Yes" else 0)

st.subheader("👤 Background Details")
age = st.number_input("Age", min_value=1, max_value=100, value=20)
gender = st.selectbox("Gender", ["m", "f"])
ethnicity = st.selectbox("Ethnicity", encoders["ethnicity"].classes_.tolist())
jaundice = st.selectbox("Jaundice at birth", ["yes", "no"])
austim = st.selectbox("Family history of ASD", ["yes", "no"])
country = st.selectbox("Country of residence", encoders["contry_of_res"].classes_.tolist())
used_app_before = st.selectbox("Used screening app before", ["yes", "no"])
relation = st.selectbox("Relation to the person", encoders["relation"].classes_.tolist())

# ---------------- Build Input for Model ----------------

input_df = pd.DataFrame([{
    "A1_Score": A_vals[0],
    "A2_Score": A_vals[1],
    "A3_Score": A_vals[2],
    "A4_Score": A_vals[3],
    "A5_Score": A_vals[4],
    "A6_Score": A_vals[5],
    "A7_Score": A_vals[6],
    "A8_Score": A_vals[7],
    "A9_Score": A_vals[8],
    "A10_Score": A_vals[9],
    "age": age,
    "gender": gender,
    "ethnicity": ethnicity,
    "jaundice": jaundice,
    "austim": austim,
    "contry_of_res": country,
    "used_app_before": used_app_before,
    "relation": relation
}])

# Encode categorical columns using training encoders
for col in input_df.columns:
    if col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

# Reorder columns to match training feature order
input_df = input_df[model.feature_names_in_]

# ---------------- Prediction ----------------

if st.button("🔍 Predict ASD", key="predict_btn"):
    
    proba = model.predict_proba(input_df)[0]
    pred = int(np.argmax(proba))
    conf = np.max(proba) * 100

    # Show ONLY model result
    if pred == 1:
        st.error("⚠️ ASD Likely (Model Prediction)")
    else:
        st.success("✅ No ASD Detected (Model Prediction)")

    st.write(f"Model Confidence: {conf:.2f}%")
    st.divider()


    # ---------------- Simple Rule-Based Fallback (as you asked) ----------------
    yes_count = sum(A_vals)
    no_count = 10 - yes_count

    st.subheader("📊 Simple Screening (Rule-based)")

    if yes_count > no_count:
        st.warning("⚠️ ASD Likely ")
    else:
        st.success("✅ No ASD Detected ")

    

    st.info("⚠️ This tool is for screening only, not a medical diagnosis.")

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = "asd-screening-secret-key-2026"

# ─── Load model and encoders ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "encoders.pkl"), "rb") as f:
    encoders = pickle.load(f)

# ─── Feature → Human-readable label map ──────────────────────────────────────
FEATURE_LABELS = {
    "A1_Score":        "Difficulty making eye contact",
    "A2_Score":        "Trouble understanding others' feelings",
    "A3_Score":        "Prefers being alone rather than with others",
    "A4_Score":        "Uncomfortable in social situations",
    "A5_Score":        "Difficulty starting or continuing conversations",
    "A6_Score":        "Takes things very literally",
    "A7_Score":        "Gets overly focused on one topic or interest",
    "A8_Score":        "Upset by small changes in routine",
    "A9_Score":        "Difficulty understanding social rules",
    "A10_Score":       "Struggles to express emotions clearly",
    "age":             "Age of the individual",
    "gender":          "Gender",
    "ethnicity":       "Ethnicity",
    "jaundice":        "Jaundice at birth",
    "austim":          "Family history of ASD",
    "contry_of_res":   "Country of residence",
    "used_app_before": "Used a screening app before",
    "relation":        "Relation to the person being screened",
}

# ─── Feature → Recommendations map ───────────────────────────────────────────
FEATURE_RECOMMENDATIONS = {
    "A1_Score": [
        "Practice short, low-pressure eye-contact moments in familiar settings.",
        "Try guided social interaction exercises with a trusted person.",
        "Use communication training programs designed for social skill building.",
    ],
    "A2_Score": [
        "Explore emotion-recognition apps or flashcard tools.",
        "Engage in role-play exercises to practise reading social cues.",
        "Consider books or videos on understanding body language and feelings.",
    ],
    "A3_Score": [
        "Gradually introduce structured group activities at a comfortable pace.",
        "Set aside short, scheduled social time with a trusted individual.",
        "Join interest-based clubs where social pressure is naturally lower.",
    ],
    "A4_Score": [
        "Start with small, familiar social gatherings before larger ones.",
        "Learn and practise relaxation techniques (deep breathing, mindfulness).",
        "Consider social-skills training with a counsellor or therapist.",
    ],
    "A5_Score": [
        "Practise conversation starters with family or close friends.",
        "Use conversation-prompt cards to ease into discussions.",
        "Seek speech-language therapy focused on pragmatic communication.",
    ],
    "A6_Score": [
        "Use concrete examples and visual aids when explaining abstract ideas.",
        "Encourage asking clarifying questions in conversations.",
        "Work with educators on teaching figurative language in context.",
    ],
    "A7_Score": [
        "Channel focused interests into structured learning or creative projects.",
        "Set gentle time boundaries for special-interest activities.",
        "Explore broader related topics to gradually widen areas of interest.",
    ],
    "A8_Score": [
        "Create consistent daily routines to reduce unexpected changes.",
        "Introduce minor planned variations to build flexibility gradually.",
        "Use visual schedules or calendars to prepare for upcoming changes.",
    ],
    "A9_Score": [
        "Use Social Stories to explain common social rules in a clear way.",
        "Practise real-life social scenarios through role-play.",
        "Engage in group social-skills classes or support groups.",
    ],
    "A10_Score": [
        "Try journaling to identify and articulate feelings regularly.",
        "Use an emotion chart or app to label and track emotions.",
        "Work with a therapist on expressive communication strategies.",
    ],
    "age": [
        "Early intervention programmes are highly effective -- seek professional assessment promptly.",
        "Age-appropriate social groups can provide safe environments to practise skills.",
    ],
    "austim": [
        "A family history of ASD means genetic counselling may be beneficial.",
        "Connect with local ASD family-support networks for shared resources.",
    ],
    "jaundice": [
        "Discuss perinatal history with a paediatrician for a comprehensive evaluation.",
        "Regular developmental check-ups are recommended for children with perinatal risk factors.",
    ],
    "gender": [
        "ASD may present differently across genders; ensure assessment tools account for this.",
    ],
    "ethnicity": [
        "Seek culturally sensitive support services and assessments in your community.",
    ],
    "contry_of_res": [
        "Look for ASD support organisations and resources available in your country.",
    ],
    "used_app_before": [
        "Consistent use of screening tools can help track changes over time.",
        "Share prior screening results with healthcare professionals for context.",
    ],
    "relation": [
        "Involve close family members in the support and therapy process.",
        "Caregiver training programmes can strengthen day-to-day support.",
    ],
}


# ─── Explainability helpers ───────────────────────────────────────────────────
def get_feature_importances(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.abs(model.coef_)
        if coef.ndim > 1:
            coef = coef[0]
        total = coef.sum()
        return coef / total if total > 0 else coef
    else:
        return np.ones(len(feature_names)) / len(feature_names)


def explain_prediction(input_df, model, feature_names, top_n=3):
    importances = get_feature_importances(model, feature_names)
    user_values = input_df.iloc[0].values
    weighted = []
    for i, fname in enumerate(feature_names):
        imp = importances[i]
        val = user_values[i]
        if fname in [f"A{j}_Score" for j in range(1, 11)]:
            weight = imp * (2.0 if val == 1 else 0.5)
        else:
            weight = imp
        weighted.append((fname, weight))
    weighted.sort(key=lambda x: x[1], reverse=True)
    return [fname for fname, _ in weighted[:top_n]]


# ─── Excel Data Logging ──────────────────────────────────────────────────────
EXCEL_PATH = os.path.join(BASE_DIR, "user_data.xlsx")

EXCEL_COLUMNS = [
    "Timestamp", "Name", "Email",
    "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
    "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
    "age", "gender", "ethnicity", "jaundice", "austim",
    "contry_of_res", "used_app_before", "relation",
    "Prediction", "Risk_Level", "ASD_Probability",
]


def save_to_excel(row: dict) -> None:
    """Append a single prediction record to user_data.xlsx."""
    try:
        new_row = pd.DataFrame([row], columns=EXCEL_COLUMNS)
        if os.path.exists(EXCEL_PATH):
            existing = pd.read_excel(EXCEL_PATH, engine="openpyxl")
            for col in EXCEL_COLUMNS:
                if col not in existing.columns:
                    existing[col] = None
            updated = pd.concat([existing, new_row], ignore_index=True)
        else:
            updated = new_row
        updated.to_excel(EXCEL_PATH, index=False, engine="openpyxl")
    except Exception as exc:
        print(f"[Excel] ERROR: {exc}")


# ─── Auth decorator ──────────────────────────────────────────────────────────
def login_required(f):
    """Redirect to /login if user is not authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_name" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ─── Routes ───────────────────────────────────────────────────────────────────

# -- Login --
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name  = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        if not name:
            return render_template("login.html", error="Please enter your name.")
        session["user_name"]  = name
        session["user_email"] = email
        return redirect(url_for("index"))
    return render_template("login.html", error=None)


# -- Logout --
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# -- Main page (protected) --
@app.route("/")
@login_required
def index():
    return render_template(
        "index.html",
        user_name=session.get("user_name", ""),
        user_email=session.get("user_email", ""),
    )


# -- API: options --
@app.route("/api/options")
@login_required
def get_options():
    return jsonify({
        "ethnicity":       encoders["ethnicity"].classes_.tolist(),
        "contry_of_res":   encoders["contry_of_res"].classes_.tolist(),
        "relation":        encoders["relation"].classes_.tolist(),
        "gender":          ["m", "f"],
        "jaundice":        ["yes", "no"],
        "austim":          ["yes", "no"],
        "used_app_before": ["yes", "no"],
    })


# -- API: predict (protected) --
@app.route("/api/predict", methods=["POST"])
@login_required
def predict():
    try:
        data = request.json

        raw = {
            "A1_Score":        int(data.get("A1_Score", 0)),
            "A2_Score":        int(data.get("A2_Score", 0)),
            "A3_Score":        int(data.get("A3_Score", 0)),
            "A4_Score":        int(data.get("A4_Score", 0)),
            "A5_Score":        int(data.get("A5_Score", 0)),
            "A6_Score":        int(data.get("A6_Score", 0)),
            "A7_Score":        int(data.get("A7_Score", 0)),
            "A8_Score":        int(data.get("A8_Score", 0)),
            "A9_Score":        int(data.get("A9_Score", 0)),
            "A10_Score":       int(data.get("A10_Score", 0)),
            "age":             float(data.get("age", 20)),
            "gender":          data.get("gender"),
            "ethnicity":       data.get("ethnicity"),
            "jaundice":        data.get("jaundice"),
            "austim":          data.get("austim"),
            "contry_of_res":   data.get("contry_of_res"),
            "used_app_before": data.get("used_app_before"),
            "relation":        data.get("relation"),
        }

        input_df = pd.DataFrame([raw])

        for col in input_df.columns:
            if col in encoders:
                input_df[col] = encoders[col].transform(input_df[col])

        input_df = input_df[model.feature_names_in_]
        feature_names = list(model.feature_names_in_)

        # Prediction
        proba    = model.predict_proba(input_df)[0]
        pred     = int(np.argmax(proba))
        conf     = float(np.max(proba) * 100)
        asd_prob = float(proba[1]) * 100

        if asd_prob <= 30:
            risk_level = "Low"
        elif asd_prob <= 70:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Log to Excel with user info from session
        excel_row = {
            "Timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Name":            session.get("user_name", ""),
            "Email":           session.get("user_email", ""),
            "A1_Score":        raw["A1_Score"],
            "A2_Score":        raw["A2_Score"],
            "A3_Score":        raw["A3_Score"],
            "A4_Score":        raw["A4_Score"],
            "A5_Score":        raw["A5_Score"],
            "A6_Score":        raw["A6_Score"],
            "A7_Score":        raw["A7_Score"],
            "A8_Score":        raw["A8_Score"],
            "A9_Score":        raw["A9_Score"],
            "A10_Score":       raw["A10_Score"],
            "age":             raw["age"],
            "gender":          raw["gender"],
            "ethnicity":       raw["ethnicity"],
            "jaundice":        raw["jaundice"],
            "austim":          raw["austim"],
            "contry_of_res":   raw["contry_of_res"],
            "used_app_before": raw["used_app_before"],
            "relation":        raw["relation"],
            "Prediction":      "ASD Likely" if pred == 1 else "No ASD Detected",
            "Risk_Level":      risk_level,
            "ASD_Probability": round(asd_prob / 100, 4),
        }
        save_to_excel(excel_row)

        # Explainability
        top_features   = explain_prediction(input_df, model, feature_names, top_n=3)
        explanations   = [FEATURE_LABELS.get(f, f) for f in top_features]

        recommendations = []
        seen = set()
        for feat in top_features:
            for tip in FEATURE_RECOMMENDATIONS.get(feat, []):
                if tip not in seen:
                    recommendations.append({
                        "trait": FEATURE_LABELS.get(feat, feat),
                        "tip":   tip,
                    })
                    seen.add(tip)

        return jsonify({
            "status":          "success",
            "prediction":      pred,
            "confidence":      conf,
            "asd_probability": asd_prob,
            "risk_level":      risk_level,
            "message":         "ASD Likely" if pred == 1 else "No ASD Detected",
            "explanations":    explanations,
            "top_features":    top_features,
            "recommendations": recommendations,
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# -- Dashboard --
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user_name=session.get("user_name", ""))


# -- API: stats --
@app.route("/api/stats")
@login_required
def get_stats():
    """Calculate aggregate statistics from user_data.xlsx for the dashboard."""
    try:
        if not os.path.exists(EXCEL_PATH):
            return jsonify({
                "total_screenings": 0,
                "risk_distribution": {"Low": 0, "Medium": 0, "High": 0},
                "avg_probability": 0,
                "recent_history": [],
                "top_traits": []
            })

        df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
        
        # Basic stats
        total = len(df)
        risk_counts = df["Risk_Level"].value_counts().to_dict()
        for level in ["Low", "Medium", "High"]:
            if level not in risk_counts:
                risk_counts[level] = 0
        
        avg_prob = float(df["ASD_Probability"].mean() * 100) if total > 0 else 0
        
        # Recent history (last 10)
        recent_df = df.tail(10)[["Timestamp", "Name", "Risk_Level", "Prediction"]]
        recent = recent_df.to_dict(orient="records")
        recent.reverse()  # Show newest first

        # Top traits in High Risk cases
        high_risk_df = df[df["Risk_Level"] == "High"]
        trait_counts = {}
        if not high_risk_df.empty:
            for i in range(1, 11):
                col = f"A{i}_Score"
                label = FEATURE_LABELS[col]
                trait_counts[label] = int(high_risk_df[col].sum())
        
        sorted_traits = sorted(trait_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_traits = [{"trait": t, "count": c} for t, c in sorted_traits]

        return jsonify({
            "total_screenings": total,
            "risk_distribution": risk_counts,
            "avg_probability": avg_prob,
            "recent_history": recent,
            "top_traits": top_traits
        })
    except Exception as e:
        print(f"[Stats API] Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400



if __name__ == "__main__":
    app.run(debug=True, port=5000)

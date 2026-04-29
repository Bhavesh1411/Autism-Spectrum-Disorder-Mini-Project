import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.units import inch

def generate_pdf():
    doc = SimpleDocTemplate("ASD_Screening_Project_Analysis.pdf", pagesize=A4,
                        rightMargin=72, leftMargin=72,
                        topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor("#2E5077"),
        alignment=1,
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor("#4DA1A9"),
        spaceBefore=20,
        spaceAfter=10
    )
    
    subheading_style = ParagraphStyle(
        'SubHeadingStyle',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor("#79D7BE"),
        spaceBefore=15,
        spaceAfter=8
    )
    
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['BodyText'],
        fontName='Courier',
        fontSize=9,
        leftIndent=20,
        textColor=colors.HexColor("#333333"),
        backColor=colors.HexColor("#F4F4F4"),
        borderPadding=5
    )
    
    body_style = styles['BodyText']
    body_style.alignment = 4  # Justified
    
    elements = []

    # --- Title Page ---
    elements.append(Spacer(1, 2*inch))
    elements.append(Paragraph("Autism Spectrum Disorder (ASD) Screening System", title_style))
    elements.append(Paragraph("Technical Analysis & Project Documentation", styles['Heading2']))
    elements.append(Spacer(1, 1*inch))
    elements.append(Paragraph("<b>Author:</b> Technical Team", styles['Normal']))
    elements.append(Paragraph("<b>Date:</b> April 29, 2026", styles['Normal']))
    elements.append(PageBreak())

    # --- Project Overview ---
    elements.append(Paragraph("1. Project Overview", heading_style))
    elements.append(Paragraph(
        "This project is a comprehensive screening tool designed to assess the risk of Autism Spectrum Disorder (ASD) "
        "using behavioral and demographic data. It leverages Machine Learning (Random Forest) to provide accurate "
        "predictions and incorporates Explainable AI (XAI) to help users understand which traits contributed most to "
        "their result.", body_style))
    
    elements.append(Paragraph("Overall Workflow:", subheading_style))
    elements.append(Paragraph(
        "1. <b>Data Collection:</b> Questionnaire-based input from users.<br/>"
        "2. <b>ML Pipeline:</b> Preprocessing, balancing data (SMOTE), and model training.<br/>"
        "3. <b>Web Application:</b> Flask-based interface for real-time predictions.<br/>"
        "4. <b>XAI:</b> Identifying top risk factors and providing actionable recommendations.<br/>"
        "5. <b>Logging:</b> Automated data archival in Excel for clinical review.", body_style))

    # --- Tables Section ---
    elements.append(Paragraph("2. Technical Specifications", heading_style))
    
    # Table 1: Technologies
    elements.append(Paragraph("Technologies Used", subheading_style))
    tech_data = [
        ['Category', 'Technology'],
        ['Backend', 'Python, Flask'],
        ['Machine Learning', 'Scikit-learn, XGBoost, SMOTE'],
        ['Data Handling', 'Pandas, NumPy, OpenPyXL'],
        ['Visualization', 'Matplotlib, Seaborn'],
        ['Frontend', 'HTML, CSS (Vanilla), Jinja2']
    ]
    t1 = Table(tech_data, colWidths=[2*inch, 3*inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2E5077")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    elements.append(t1)
    
    # Table 2: Library Versions
    elements.append(Paragraph("Library Versions", subheading_style))
    lib_data = [
        ['Library', 'Version'],
        ['Flask', '3.0.0'],
        ['Pandas', '2.1.4'],
        ['Scikit-learn', '1.3.2'],
        ['XGBoost', '2.0.3'],
        ['Numpy', '1.26.4'],
        ['Imbalanced-learn', '0.11.0']
    ]
    t2 = Table(lib_data, colWidths=[2*inch, 3*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4DA1A9")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(t2)
    elements.append(PageBreak())

    # --- ML.PY ANALYSIS ---
    elements.append(Paragraph("3. Machine Learning Pipeline Analysis (ml.py)", heading_style))
    elements.append(Paragraph("Phase 1: Data Preparation", subheading_style))
    elements.append(Paragraph("The pipeline starts by loading the raw dataset and performing initial cleanup.", body_style))
    elements.append(Paragraph(
        "<b>Code Snippet:</b><br/>"
        "df = pd.read_csv('train.csv')<br/>"
        "df = df.drop(columns=['ID', 'age_desc', 'result'])", code_style))
    elements.append(Paragraph(
        "<b>Explanation:</b><br/>"
        "- <b>Lines 16-17:</b> Loads the training data into a Pandas DataFrame.<br/>"
        "- <b>Line 23:</b> Drops columns that either cause data leakage (result) or are irrelevant for prediction (ID, age_desc).", body_style))

    elements.append(Paragraph("Phase 2: Encoding & Preprocessing", subheading_style))
    elements.append(Paragraph(
        "<b>Code Snippet:</b><br/>"
        "for column in object_columns:<br/>"
        "&nbsp;&nbsp;le = LabelEncoder()<br/>"
        "&nbsp;&nbsp;df[column] = le.fit_transform(df[column])", code_style))
    elements.append(Paragraph(
        "<b>Explanation:</b><br/>"
        "- <b>Lines 59-64:</b> Initializes a LabelEncoder for each categorical column. This converts text data (e.g., 'm'/'f') into numbers (0/1).<br/>"
        "- <b>Lines 67-68:</b> Saves these encoders into a pickle file so the Flask app can use the exact same mapping for real-time user inputs.", body_style))

    elements.append(Paragraph("Phase 3: Class Balancing (SMOTE)", subheading_style))
    elements.append(Paragraph(
        "<b>Explanation:</b><br/>"
        "- <b>Lines 112-113:</b> ASD datasets are often imbalanced (fewer positive cases). SMOTE generates synthetic samples for the minority class to ensure the model doesn't become biased towards 'No ASD'.", body_style))

    elements.append(Paragraph("Phase 4: Training & Hyperparameter Tuning", subheading_style))
    elements.append(Paragraph(
        "<b>Explanation:</b><br/>"
        "- <b>Lines 119-123:</b> Tests multiple models: Decision Tree, Random Forest, and XGBoost.<br/>"
        "- <b>Lines 178-186:</b> Uses RandomizedSearchCV to find the best configuration for Random Forest (n_estimators, max_depth, etc.).", body_style))
    elements.append(PageBreak())

    # --- APP_FLASK.PY ANALYSIS ---
    elements.append(Paragraph("4. Flask Application Analysis (app_flask.py)", heading_style))
    elements.append(Paragraph("Integration & Initialization", subheading_style))
    elements.append(Paragraph(
        "<b>Code Snippet:</b><br/>"
        "with open('best_model.pkl', 'rb') as f: model = pickle.load(f)", code_style))
    elements.append(Paragraph(
        "<b>Explanation:</b><br/>"
        "- <b>Lines 15-19:</b> Loads the trained model and encoders at server startup. This allows the app to perform sub-second predictions without retraining.", body_style))

    elements.append(Paragraph("The Prediction Route", subheading_style))
    elements.append(Paragraph(
        "<b>Code Snippet:</b><br/>"
        "@app.route('/api/predict', methods=['POST'])<br/>"
        "def predict():<br/>"
        "&nbsp;&nbsp;data = request.json<br/>"
        "&nbsp;&nbsp;input_df = pd.DataFrame([data])<br/>"
        "&nbsp;&nbsp;proba = model.predict_proba(input_df)[0]", code_style))
    elements.append(Paragraph(
        "<b>Explanation:</b><br/>"
        "- <b>Lines 252-273:</b> Captures the JSON data sent from the frontend.<br/>"
        "- <b>Line 279:</b> Uses the pre-loaded encoders to transform the user's input.<br/>"
        "- <b>Line 285:</b> Calculates the probability of ASD.<br/>"
        "- <b>Lines 290-295:</b> Assigns a Risk Level (Low/Medium/High) based on the calculated probability percentage.", body_style))

    elements.append(Paragraph("Explainable AI (XAI) Logic", subheading_style))
    elements.append(Paragraph(
        "<b>Explanation:</b><br/>"
        "- <b>Lines 141-154:</b> The 'explain_prediction' function calculates which features had the highest weight in the specific prediction. "
        "It looks at feature importance and user input values to identify the top 3 'risk contributors'.", body_style))
    elements.append(PageBreak())

    # --- Model Selection ---
    elements.append(Paragraph("5. Model Selection & Justification", heading_style))
    elements.append(Paragraph(
        "<b>Models Used:</b> Decision Tree, Random Forest, XGBoost.<br/>"
        "<b>Winner:</b> Random Forest Classifier.", subheading_style))
    
    elements.append(Paragraph(
        "<b>Why Random Forest?</b><br/>"
        "1. <b>Ensemble Learning:</b> By combining multiple Decision Trees, it significantly reduces the risk of overfitting.<br/>"
        "2. <b>Non-Linearity:</b> ASD traits often have complex, non-linear relationships which Random Forest captures better than simple linear models.<br/>"
        "3. <b>Robustness:</b> It handles noise and outliers in behavioral data very effectively.<br/>"
        "4. <b>Feature Importance:</b> It provides built-in mechanisms to understand feature significance, enabling our XAI features.", body_style))

    elements.append(Paragraph(
        "<b>Why not alternatives?</b><br/>"
        "- <b>Decision Trees:</b> Too prone to overfitting the training data.<br/>"
        "- <b>Linear Regression:</b> Ineffective because the problem is classification and non-linear.<br/>"
        "- <b>SVM:</b> Harder to interpret for XAI and computationally more expensive for this dataset size.", body_style))
    elements.append(PageBreak())

    # --- Interview Q&A ---
    elements.append(Paragraph("6. Interview Q&A: Top 30 Questions", heading_style))
    
    qa_list = [
        ("1. What is the core objective of this system?", "To provide a preliminary, model-driven screening tool for ASD based on behavioral traits and demographic data."),
        ("2. Why did you choose Random Forest over a simpler model?", "Random Forest handles the non-linear complexity of behavioral traits better and reduces overfitting through ensemble bagging."),
        ("3. What is SMOTE and why is it used?", "Synthetic Minority Over-sampling Technique. It addresses class imbalance by creating synthetic instances of the minority class (ASD positive)."),
        ("4. How does your app ensure 'Explainability'?", "We use feature importance from the model to identify which traits (A1-A10 scores) contributed most to a specific user's prediction."),
        ("5. Why are columns like 'result' dropped in ml.py?", "The 'result' column is usually a simple sum of scores. Including it would cause 'data leakage', making the model learn a trivial rule instead of complex patterns."),
        ("6. What is the significance of the 70/15/15 split?", "It provides enough data for training (70%), a separate set for monitoring progress/tuning (15% validation), and a final set for unbiased evaluation (15% test)."),
        ("7. How do you handle missing values ('?')?", "We replace them with 'Others' for categorical data like ethnicity, ensuring the model can still process the record without losing information."),
        ("8. What is 'Confidence' in your prediction API?", "It is the probability returned by the model for the predicted class (e.g., if the model is 85% sure the class is ASD)."),
        ("9. Why use Flask instead of a static page?", "Flask allows us to run Python code (our model) on the backend to process user inputs in real-time."),
        ("10. What is a 'Confusion Matrix' and why is it important?", "It shows True Positives, True Negatives, False Positives, and False Negatives, helping us see if the model is confusing the two classes."),
        ("11. How do you prevent overfitting?", "By using ensemble models (Random Forest), cross-validation, and hyperparameter tuning."),
        ("12. What is the role of LabelEncoder?", "It transforms categorical strings into integers so the mathematical algorithms of ML can process them."),
        ("13. Why log data to Excel?", "For historical tracking, clinical audit, and allowing experts to review automated screenings later."),
        ("14. What are 'A1-A10' scores?", "Standardized behavioral screening parameters used globally in autism assessment protocols."),
        ("15. Can this replace a doctor's diagnosis?", "No. It is a screening tool designed to encourage individuals to seek professional consultation if risk is detected."),
        ("16. How does the app handle security?", "It uses a secret key for session management and basic auth decorators to protect the main screening pages."),
        ("17. What is 'Cross-Validation'?", "A technique where the data is split into N parts; the model is trained on N-1 and tested on the 1 part, repeating N times to ensure stable accuracy."),
        ("18. Why use XGBoost?", "It's a powerful gradient boosting method that often provides higher accuracy but requires more tuning than Random Forest."),
        ("19. What is 'Early Stopping' in XGBoost?", "It stops training once the validation error stops improving, preventing the model from learning noise."),
        ("20. How is 'Risk Level' calculated?", "By mapping the model's output probability: 0-30% is Low, 31-70% is Medium, and 71-100% is High Risk."),
        ("21. Why save encoders as pickle files?", "To ensure that the mapping used during training (e.g., 'm' = 0) is identical to the one used in production."),
        ("22. What is the impact of Jaundice at birth?", "Perinatal factors like jaundice are sometimes correlated with developmental disorders, and the model evaluates this weight."),
        ("23. How would you handle a larger dataset?", "I would migrate the Excel logging to a SQL database like PostgreSQL for better scalability and concurrency."),
        ("24. What is the 'F1-Score' in your report?", "It's the harmonic mean of Precision and Recall, providing a single metric that balances both."),
        ("25. Why did you use 'RandomizedSearchCV'?", "It is faster than GridSearchCV as it samples a fixed number of parameter combinations instead of checking every single one."),
        ("26. What are recommendations based on?", "They are mapped directly to the top contributing factors identified by the XAI logic."),
        ("27. How does SMOTE differ from simple oversampling?", "Simple oversampling duplicates records; SMOTE creates new, synthetic records that are mathematically similar to the existing ones."),
        ("28. What is 'Data Leakage'?", "When information from outside the training dataset is used to create the model, leading to over-optimistic results."),
        ("29. Why use 'Agg' backend for Matplotlib?", "It allows generating plots in scripts without requiring a GUI window, which is essential for server-side or automated tasks."),
        ("30. How would you improve this project next?", "By adding a deep learning model for comparison or integrating SHAP values for even more detailed explainability.")
    ]

    for q, a in qa_list:
        elements.append(Paragraph(f"<b>Q: {q}</b>", body_style))
        elements.append(Paragraph(f"A: {a}", body_style))
        elements.append(Spacer(1, 10))

    elements.append(PageBreak())
    elements.append(Paragraph("End of Documentation", title_style))

    doc.build(elements)
    print("PDF generated: ASD_Screening_Project_Analysis.pdf")

if __name__ == "__main__":
    generate_pdf()

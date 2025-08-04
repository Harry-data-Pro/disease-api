import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("model/disease_model.pkl")
label = joblib.load("model/label_encoder.pkl")

# List of all 132 symptom features (from your notebook/dataset; add all if needed)
FEATURES = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering",
    "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue",
    # ... Add the full list of 132 symptoms here (e.g., from your df.columns in the notebook)
    # For brevity, I'm truncating; replace with your actual list.
    "blackheads", "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails",
    "inflammatory_nails", "blister", "red_sore_around_nose", "yellow_crust_ooze"
]  # Ensure this matches your model's expected features

# Streamlit app
st.title("Disease Predictor")

st.write("Select your symptoms below and click 'Predict' to get a diagnosis.")

# Create a form for symptoms
with st.form(key="symptom_form"):
    selected_symptoms = {}
    for symptom in FEATURES:
        selected_symptoms[symptom] = st.checkbox(symptom.capitalize().replace("_", " "))

    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    # Build the feature vector (0/1 Series)
    row = pd.Series(0, index=FEATURES, dtype=int)
    for symptom, value in selected_symptoms.items():
        row[symptom] = 1 if value else 0

    # Predict
    prob = model.predict_proba([row])[0]
    idx = prob.argmax()
    disease = label.inverse_transform([idx])[0]
    probability = float(prob[idx])

    # Display result
    st.success(f"Predicted Disease: **{disease}**")
    st.info(f"Probability: **{probability:.2%}**")
    st.write("Note: This is a model prediction—consult a doctor for real advice!")

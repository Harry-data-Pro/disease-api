import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("model/disease_model.pkl")
label = joblib.load("model/label_encoder.pkl")

# Dynamically get the exact list of features from the model (should be 132)
FEATURES = sorted(list(model.feature_name_))

# Streamlit app
st.title("Disease Predictor")

st.write("Select your symptoms below and click 'Predict' to get a diagnosis.")

# Create a form for symptoms
with st.form(key="symptom_form"):
    selected_symptoms = {}
    for symptom in FEATURES:
        # Display symptom nicely (e.g., "Skin Rash" instead of "skin_rash")
        display_name = symptom.capitalize().replace("_", " ")
        selected_symptoms[symptom] = st.checkbox(display_name)

    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    # Build the feature vector (0/1 Series with ALL features)
    row = pd.Series(0, index=FEATURES, dtype=int)
    for symptom, selected in selected_symptoms.items():
        if selected:
            row[symptom] = 1  # Set to 1 only if checked

    # Predict
    prob = model.predict_proba([row])[0]
    idx = prob.argmax()
    disease = label.inverse_transform([idx])[0]
    probability = float(prob[idx])

    # Display result
    st.success(f"Predicted Disease: **{disease}**")
    st.info(f"Probability: **{probability:.2%}**")
    st.write("Note: This is a model prediction—consult a doctor for real advice!")


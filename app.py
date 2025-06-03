import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any

# Set page config
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="centered"
)

# Define the expected feature order (matching the training data)
FEATURE_ORDER = [
    'gender', 'age', 'hypertension', 'heart_disease', 'bmi',
    'HbA1c_level', 'blood_glucose_level', 'HighChol', 'Stroke', 'HeartDiseaseorAttack'
]

# Load the saved model, scaler, and PCA
@st.cache_resource
def load_models() -> Dict[str, Any]:
    return {
        'model': joblib.load('models/random_forest_model.joblib'),
        'scaler': joblib.load('models/scaler.joblib'),
        'pca': joblib.load('models/pca.joblib')
    }

# Function to make prediction
def predict_diabetes(input_data: pd.DataFrame, models: Dict[str, Any]) -> int:
    # Ensure the input data has the correct feature order
    input_data = input_data[FEATURE_ORDER]
    
    # Scale the input data
    scaled_data = models['scaler'].transform(input_data)
    
    # Apply PCA
    pca_data = models['pca'].transform(scaled_data)
    
    # Make prediction
    prediction = models['model'].predict(pca_data)
    return prediction[0]

# Main app
def main():
    st.title("🏥 Diabetes Prediction Model")
    st.write("""
    This app predicts the likelihood of diabetes based on various health indicators.
    Please fill in the patient's information below.
    """)

    # Create two columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7, step=0.01)
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.1)
        hba1c = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.7, step=0.1)
        glucose = st.number_input("Blood Glucose Level", min_value=0, max_value=500, value=100)

    with col2:
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        high_cholesterol = st.selectbox("High Cholesterol", ["No", "Yes"])
        stroke = st.selectbox("Stroke History", ["No", "Yes"])
        heart_attack = st.selectbox("Heart Attack History", ["No", "Yes"])

    # Calculate BMI
    bmi = weight / (height ** 2)

    # Convert categorical variables to binary
    gender_binary = 1 if gender == "Male" else 0
    hypertension_binary = 1 if hypertension == "Yes" else 0
    heart_disease_binary = 1 if heart_disease == "Yes" else 0
    high_cholesterol_binary = 1 if high_cholesterol == "Yes" else 0
    stroke_binary = 1 if stroke == "Yes" else 0
    heart_attack_binary = 1 if heart_attack == "Yes" else 0

    # Create input dataframe with the correct feature order
    input_data = pd.DataFrame({
        'gender': [gender_binary],
        'age': [age],
        'hypertension': [hypertension_binary],
        'heart_disease': [heart_disease_binary],
        'bmi': [bmi],
        'HbA1c_level': [hba1c],
        'blood_glucose_level': [glucose],
        'HighChol': [high_cholesterol_binary],
        'Stroke': [stroke_binary],
        'HeartDiseaseorAttack': [heart_attack_binary]
    })

    # Load models
    models = load_models()

    # Make prediction when button is clicked
    if st.button("Predict Diabetes Risk"):
        prediction = predict_diabetes(input_data, models)
        
        # Display result
        st.markdown("---")
        if prediction == 1:
            st.error("⚠️ The model predicts a HIGH risk of diabetes.")
        else:
            st.success("✅ The model predicts a LOW risk of diabetes.")
        
        # Display additional information
        st.markdown("### Health Metrics Summary")
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric("BMI", f"{bmi:.1f}")
            st.metric("HbA1c Level", f"{hba1c:.1f}")
            st.metric("Blood Glucose", f"{glucose}")
        
        with metrics_col2:
            st.metric("Age", age)
            st.metric("Gender", gender)
            st.metric("Hypertension", hypertension)

if __name__ == "__main__":
    main() 
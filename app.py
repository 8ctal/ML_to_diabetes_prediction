import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Set page config
st.set_page_config(
    page_title="Diabetes Prediction Model",
    page_icon="🏥",
    layout="centered"
)

# Title and description
st.title("Diabetes Prediction Model")
st.markdown("""
This application predicts the probability of diabetes based on patient data.
Please fill in the patient information below.
""")

# Load the model and scaler
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('dnn_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# Create input fields
st.subheader("Patient Information")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

with col2:
    hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.7, step=0.1)
    glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)
    high_chol = st.selectbox("High Cholesterol", ["No", "Yes"])
    stroke = st.selectbox("Stroke History", ["No", "Yes"])
    heart_attack = st.selectbox("Heart Attack History", ["No", "Yes"])

# Convert categorical variables
gender = 1 if gender == "Male" else 0
hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0
high_chol = 1 if high_chol == "Yes" else 0
stroke = 1 if stroke == "Yes" else 0
heart_attack = 1 if heart_attack == "Yes" else 0

# Create predict button
if st.button("Predict Diabetes Risk"):
    try:
        # Load model and scaler
        model, scaler = load_resources()
        
        # Prepare input data
        input_data = np.array([[
            age, gender, hypertension, heart_disease, bmi,
            hba1c, glucose, high_chol, stroke, heart_attack
        ]])
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled, verbose=0)
        probability = float(prediction[0][0])  # Convert to Python float
        
        # Display results
        st.subheader("Prediction Results")
        
        # Create a progress bar with the probability
        st.progress(probability)
        
        # Display the probability as a percentage
        st.write(f"Probability of Diabetes: {probability:.1%}")
        
        # Determine risk level with adjusted thresholds
        if probability < 0.2:
            risk_level = "Very Low"
            color = "green"
            interpretation = "Normal glucose metabolism"
        elif probability < 0.4:
            risk_level = "Low"
            color = "lightgreen"
            interpretation = "Normal glucose metabolism with some risk factors"
        elif probability < 0.6:
            risk_level = "Moderate"
            color = "orange"
            interpretation = "Prediabetes range"
        elif probability < 0.8:
            risk_level = "High"
            color = "darkorange"
            interpretation = "High risk of diabetes"
        else:
            risk_level = "Very High"
            color = "red"
            interpretation = "Very high risk of diabetes"
        
        st.markdown(f'<p style="color: {color}; font-size: 20px;">Risk Level: {risk_level}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: {color};">Interpretation: {interpretation}</p>', unsafe_allow_html=True)
        
        # Display recommendations based on risk level
        st.subheader("Recommendations")
        if risk_level == "Very Low":
            st.write("""
            - Continue with your current healthy lifestyle
            - Maintain regular annual check-ups
            - Keep monitoring your blood sugar levels annually
            """)
        elif risk_level == "Low":
            st.write("""
            - Continue with your current healthy lifestyle
            - Schedule regular check-ups every 6 months
            - Monitor blood sugar levels every 6 months
            - Consider lifestyle modifications if you have other risk factors
            """)
        elif risk_level == "Moderate":
            st.write("""
            - Schedule an appointment with your healthcare provider
            - Implement lifestyle modifications:
              * Increase physical activity
              * Follow a balanced diet
              * Maintain a healthy weight
            - Monitor blood sugar levels every 3 months
            - Consider preventive measures
            """)
        elif risk_level == "High":
            st.write("""
            - Schedule an immediate appointment with your healthcare provider
            - Implement strict lifestyle modifications:
              * Regular exercise program
              * Strict dietary control
              * Weight management
            - Monitor blood sugar levels monthly
            - Consider preventive medication if prescribed
            """)
        else:  # Very High
            st.write("""
            - Schedule an urgent appointment with your healthcare provider
            - Implement immediate lifestyle changes:
              * Daily exercise routine
              * Strict dietary control
              * Weight management
            - Monitor blood sugar levels weekly
            - Follow medical advice for preventive measures
            - Consider regular screening for diabetes complications
            """)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please make sure all the model files are properly loaded.")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>This is a prediction model and should not be used as a substitute for professional medical advice.</p>
    <p>Always consult with healthcare professionals for proper diagnosis and treatment.</p>
</div>
""", unsafe_allow_html=True) 
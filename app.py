import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")

@st.cache_resource
def load_assets():
    model = joblib.load('LR_heart.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.title("🫀 Heart Disease Risk Prediction")
st.write("Input clinical features below to evaluate the likelihood of heart disease.")

# Input Layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=['M', 'F'])
    cp = st.selectbox("Chest Pain Type", options=['ATA', 'NAP', 'ASY', 'TA'])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=0, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", options=[0, 1])

with col2:
    restecg = st.selectbox("Resting ECG Results", options=['Normal', 'ST', 'LVH'])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=['N', 'Y'])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=-5.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("ST Slope", options=['Up', 'Flat', 'Down'])

if st.button("Predict"):
    # 1. Feature Mapping for Categorical Columns
    is_Male = 1 if sex == 'M' else 0
    ChestPainType_ATA = 1 if cp == 'ATA' else 0
    ChestPainType_NAP = 1 if cp == 'NAP' else 0
    ChestPainType_TA = 1 if cp == 'TA' else 0
    RestingECG_Normal = 1 if restecg == 'Normal' else 0
    RestingECG_ST = 1 if restecg == 'ST' else 0
    ExerciseAngina_Y = 1 if exang == 'Y' else 0
    ST_Slope_Flat = 1 if slope == 'Flat' else 0
    ST_Slope_Up = 1 if slope == 'Up' else 0

    # 2. FIX: Pre-scale the continuous variables before applying scaler.pkl
    # The saved scaler.pkl was fitted on ALREADY SCALED continuous variables.
    # We must replicate the first scaling step using the original heart.csv statistics.
    age_scaled = (age - 53.510893) / 9.427478
    trestbps_scaled = (trestbps - 132.396514) / 18.504067
    chol_scaled = (chol - 198.799564) / 109.324551
    thalach_scaled = (thalach - 136.809368) / 25.446463
    oldpeak_scaled = (oldpeak - 0.887364) / 1.065989

    # 3. Build the DataFrame exactly matching the order in scaler.pkl
    feature_columns = [
        'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 
        'is_Male', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 
        'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up'
    ]
    
    input_df = pd.DataFrame([[
        age_scaled, trestbps_scaled, chol_scaled, fbs, thalach_scaled, oldpeak_scaled,
        is_Male, ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA,
        RestingECG_Normal, RestingECG_ST, ExerciseAngina_Y, ST_Slope_Flat, ST_Slope_Up
    ]], columns=feature_columns)
    
    # 4. Final Scale and Predict
    final_scaled_data = scaler.transform(input_df)
    
    # Using decision_function & sigmoid mathematically to avoid predict_proba version conflicts
    score = model.decision_function(final_scaled_data)[0]
    probability = 1 / (1 + np.exp(-score))
    prediction = 1 if score > 0 else 0
    
    if prediction == 1:
        st.error("High risk of heart disease detected. Please consult a medical professional.")
    else:
        st.success("Low risk of heart disease.")
        
    st.info(f"Model Confidence (Probability of Disease): **{probability:.2%}**")
import streamlit as st
import pandas as pd
import numpy as np
import pickle

LOGO_URL = "https://i.postimg.cc/vmd5cHQP/Whats-App-Image-2026-11-08-at-9-51-29-PM.jpg"

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon=LOGO_URL,
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

artifacts = load_artifacts()

num_imputer = artifacts['num_imputer']
cat_imputer = artifacts['cat_imputer']
scaler = artifacts['scaler']
encoder = artifacts['encoder']
numerical_cols = artifacts['numerical_cols']
categorical_cols = artifacts['categorical_cols']
model = artifacts['model']

st.sidebar.image(LOGO_URL, width=200)
st.sidebar.header("Input Features")

st.title("Student Academic Performance Predictor")
st.write("Developer: Mohammed Anowar")

hours_studied = st.sidebar.slider("Hours Studied", 0, 44, 20)
attendance = st.sidebar.slider("Attendance Rate (%)", 60, 100, 85)
sleep_hours = st.sidebar.slider("Sleep Hours", 4, 10, 7)
previous_scores = st.sidebar.slider("Previous Scores", 40, 100, 70)
tutoring_sessions = st.sidebar.slider("Tutoring Sessions", 0, 8, 2)
physical_activity = st.sidebar.slider("Physical Activity", 0, 6, 2)

parental_involvement = st.sidebar.selectbox("Parental Involvement", ["Low", "Medium", "High"])
access_to_resources = st.sidebar.selectbox("Access to Resources", ["Low", "Medium", "High"])
extracurricular = st.sidebar.selectbox("Extracurricular Activities", ["Yes", "No"])
motivation = st.sidebar.selectbox("Motivation Level", ["Low", "Medium", "High"])
internet = st.sidebar.selectbox("Internet Access", ["Yes", "No"])
family_income = st.sidebar.selectbox("Family Income", ["Low", "Medium", "High"])
teacher_quality = st.sidebar.selectbox("Teacher Quality", ["Low", "Medium", "High"])
school_type = st.sidebar.selectbox("School Type", ["Public", "Private"])
peer_influence = st.sidebar.selectbox("Peer Influence", ["Positive", "Neutral", "Negative"])
learning_disabilities = st.sidebar.selectbox("Learning Disabilities", ["Yes", "No"])
parental_education = st.sidebar.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
distance = st.sidebar.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

input_dict = {
    'Hours_Studied': hours_studied,
    'Attendance': attendance,
    'Sleep_Hours': sleep_hours,
    'Previous_Scores': previous_scores,
    'Tutoring_Sessions': tutoring_sessions,
    'Physical_Activity': physical_activity,
    'Parental_Involvement': parental_involvement,
    'Access_to_Resources': access_to_resources,
    'Extracurricular_Activities': extracurricular,
    'Motivation_Level': motivation,
    'Internet_Access': internet,
    'Family_Income': family_income,
    'Teacher_Quality': teacher_quality,
    'School_Type': school_type,
    'Peer_Influence': peer_influence,
    'Learning_Disabilities': learning_disabilities,
    'Parental_Education_Level': parental_education,
    'Distance_from_Home': distance,
    'Gender': gender
}

input_df = pd.DataFrame([input_dict])

st.subheader("Selected Student Details")
st.dataframe(input_df)

if st.button("Predict Student Outcome"):
    num_data = num_imputer.transform(input_df[numerical_cols])
    num_scaled = scaler.transform(num_data)
    
    cat_data = cat_imputer.transform(input_df[categorical_cols])
    cat_encoded = encoder.transform(cat_data)
    
    processed_input = np.hstack((num_scaled, cat_encoded))
    
    prediction = model.predict(processed_input)[0]
    proba = model.predict_proba(processed_input)[0]
    
    if prediction == 1:
        st.success(f"PASS (Probability: {proba[1]*100:.2f}%)")
    else:
        st.error(f"FAIL (Probability: {proba[0]*100:.2f}%)")

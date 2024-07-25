import streamlit as st
import pickle
import numpy as np

# Load the trained model from the pkl file
with open('Myclf.pkl', 'rb') as file:
    clf = pickle.load(file)

# Title and description
st.title("Heart Disease Prediction")
st.write(
    """
    This app predicts whether a patient has heart disease or not based on their medical attributes.
    Enter the patient's details below:
    """
)

# Input fields for user to enter data
age = st.number_input("Age", min_value=0, max_value=120, value=25)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, value=120)
chol = st.number_input("Serum Cholesterol in mg/dl", min_value=0, max_value=600, value=150)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Flourosopy", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])

# Button to predict
if st.button("Predict"):
    # Create a feature array for the model
    features = np.array([
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]).reshape(1, -1)

    # Make prediction
    prediction = clf.predict(features)[0]

    # Display the result
    if prediction == 0:
        st.success("Heart Disease NOT Detected")
    else:
        st.error("Heart Disease Detected")


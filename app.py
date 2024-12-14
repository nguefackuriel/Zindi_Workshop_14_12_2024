import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd






# Load my model 

model = joblib.load('best_xgb_model.joblib')


# Styreamlit

st.title('Diabetes Prediction App')

st.write('Enter the following details to predict diabetes')

# Features

Pregnancies = st.number_input("Pregnancies", 0, 20)

glucose = st.number_input("Glucose", 0, 200)

BloodPressure = st.number_input("BloodPressure", 0, 200)

SkinThickness = st.number_input("SkinThickness", 0, 200)

Insulin = st.number_input("Insulin", 0, 500)

BMI = st.number_input("BMI", 0, 70)

DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", 0, 5)

Age = st.number_input("Age", 0, 120)



x_train = pd.read_csv('X_train.csv')

scaler = StandardScaler()

X_train = scaler.fit_transform(x_train)



#prepare input for prediction

input_data = np.array([[Pregnancies, glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

input_data = scaler.transform(input_data)


if st.button('Diabete test result'):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.write("The model predicts that this patient  **has diabetes**")
    else:
        st.write("The model predicts that this patient  **does not have diabetes**")
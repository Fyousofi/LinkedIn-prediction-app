#!/usr/bin/env python
# coding: utf-8

# # Final Project
# ## Fatima Yousofi
# ### December 10, 2024

# ***Part 2: Deploying the model on Streamlit***

import streamlit as st
from PIL import Image
import joblib  # For loading the pre-trained model
import numpy as np

# Streamlit App Title and Description
st.title("LinkedIn User Prediction App")
st.write("This app predicts whether an individual is a LinkedIn user based on demographic and behavioral factors.")

# Load the pre-trained model
model = joblib.load("model.pkl")  # Ensure 'model.pkl' is in the same directory as your app
st.success("Model loaded successfully!")  # Optional message to confirm loading

#  #### Interactive inputs for the App
st.sidebar.header("Input Features")

income = st.sidebar.slider("Income (1 = Low, 9 = High):", min_value=1, max_value=9, value=5)
education = st.sidebar.slider("Education Level (1 = Low, 8 = High):", min_value=1, max_value=8, value=4)
#parent = st.sidebar.selectbox("Parent Status: (Do you Kids)", options={0: "Not a Parent", 1: "Parent"})
#marital_status = st.sidebar.selectbox("Marital Status:", options={0: "Not Married", 1: "Married"})
age = st.sidebar.slider("Age:", min_value=18, max_value=98, value=30)

# Parent Status (Yes/No)
parent_status = st.sidebar.selectbox("Are you a parent?", options=["No", "Yes"])
parent = 1 if parent_status == "Yes" else 0

# Marital Status (Yes/No)
marital_status_input = st.sidebar.selectbox("Are you married?", options=["No", "Yes"])
marital_status = 1 if marital_status_input == "Yes" else 0
#gender = st.sidebar.selectbox("Gender:", options={0: "Male", 1: "Female"})

 # Gender Selection with Images in Sidebar
st.sidebar.write("### Select Gender")
gender_options = {
    "Male": Image.open("male.png"),  # Ensure these files are in your project folder
    "Female": Image.open("female.png")
    }
gender = st.sidebar.radio(
    "Choose Gender:",
    options=list(gender_options.keys()),
    format_func=lambda x: f"{x}"  # Customize display text for each option if needed
    )
gender_value = 0 if gender == "Male" else 1

# Display the selected image next to the radio button
st.sidebar.image(gender_options[gender], caption=gender, use_container_width=True)

 # Instruction Message in Smaller Font
st.markdown("### Submit Your Inputs")
st.markdown(
    "<p style='font-size:14px;'>Press the <strong>Submit</strong> button to see the prediction results.</p>",
        unsafe_allow_html=True,
)

# Submit Button on Main Screen
submit = st.button("Submit")

if submit:
# Prepare input for prediction
    input_data = [[income, education, parent, marital_status, gender_value, age]]

    # Predict using the model
    prediction_probability = model.predict_proba(input_data)[0][1]
    prediction = "LinkedIn User" if prediction_probability >= 0.5 else "Not a LinkedIn User"

    # Display Results
    st.write("### Prediction Results")
        
    # Display the prediction as text
    if prediction == "LinkedIn User":
        st.success(f"🎉 You are predicted to be a LinkedIn user!")
    else:
        st.error(f"❌ You are predicted NOT to be a LinkedIn user.")
        
    # Probability Strength Bar
    st.write("#### Probability Strength")
    st.progress(int(prediction_probability * 100))  # Converts probability to percentage and displays as a progress bar
        
    # Display the exact probability as a metric
    st.metric(label="LinkedIn User Probability", value=f"{prediction_probability:.2%}")

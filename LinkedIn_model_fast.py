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

# Income Dropdown
income_options = {
    1: "Less than $10,000",
    2: "$10,000 to under $20,000",
    3: "$20,000 to under $30,000",
    4: "$30,000 to under $40,000",
    5: "$40,000 to under $50,000",
    6: "$50,000 to under $75,000",
    7: "$75,000 to under $100,000",
    8: "$100,000 to under $150,000",
    9: "$150,000 or more",
}
income = st.sidebar.selectbox(
    "Household Income:",
    options=list(income_options.keys()),
    format_func=lambda x: income_options[x]
)


# Education Dropdown
education_options = {
    1: "Less than high school",
    2: "High school incomplete",
    3: "High school graduate",
    4: "Some college, no degree",
    5: "Two-year associate degree",
    6: "Four-year college degree",
    7: "Some postgraduate schooling",
    8: "Postgraduate or professional degree"
}
education = st.sidebar.selectbox(
    "Education Level:",
    options=list(education_options.keys()),
    format_func=lambda x: education_options[x]
)
age = st.sidebar.slider("Age:", min_value=18, max_value=98, value=30)

# Parent Status (Yes/No)
parent_status = st.sidebar.selectbox("Are you a parent of a child under 18 living in your home?", options=["No", "Yes"])
parent = 1 if parent_status == "Yes" else 0

# Marital Status (Yes/No)
marital_status_input = st.sidebar.selectbox("Are you married?", options=["No", "Yes"])
marital_status = 1 if marital_status_input == "Yes" else 0

gender_input = st.sidebar.selectbox("Gender:", options=["Female", "Male / Other"])
gender_value = 1 if gender_input == "Female" else 0

#gender = st.sidebar.selectbox("Gender:", options={0: "Male/Other", 1: "Female"})


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

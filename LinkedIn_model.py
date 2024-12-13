#!/usr/bin/env python
# coding: utf-8

# # Final Project
# ## Fatima Yousofi
# ### December 10, 2024

# ***Part 1: Building a classification model to predict LinkedIn users***

# In[3]:

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import os
import numpy as np
from sklearn.model_selection import train_test_split #used for splitting the dataset for modeling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# #### Q1: Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

# In[1]:


file_path = r"C:\Users\mo\Desktop\fatima\LindedIn_PredictionApp\social_media_usage.csv"


# In[6]:


s = pd.read_csv(file_path)
print(s.shape)  #Checking dataset dimensions


# #### Q2: Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

# In[8]:


def clean_sm(x):
    return np.where(x == 1, 1, 0)
#Testing clean_sm
Toy_df = pd.DataFrame({'col1': [1, 0, 3], 'col2': [0, 1, 1]})
Toy_df['cleaned'] = Toy_df['col1'].apply(clean_sm)
print(Toy_df)


# #### Q3:Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

# In[10]:


ss = s[['income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()
ss['sm_li'] = clean_sm(s['web1h'])  # Replace 'LinkedIn' with actual column name from the variable dictionary

#Drop missing values
ss = ss[(ss['income'] <= 9) & (ss['educ2'] <= 8) & (ss['age'] <= 98)]
ss.dropna(inplace=True)

#Converting Parent, marital and gender to binary variables for further analysis
ss['par'] = ss['par'].map({1: 0, 2: 1})  # Assuming 1 = non-parent, 2 = parent
ss['marital'] = ss['marital'].apply(lambda x: 1 if x == 1 else 0)  # Assuming 1 = married, others = not married
ss['gender'] = ss['gender'].apply(lambda x: 1 if x == 1 else 0)  # Assuming 1 = female, others = male

print("Summary of the new dataframe `ss`:")
print(ss.describe()) #come back to this for double checking


# **Performing Exploratory Data Analysis**

# In[12]:


#Target variable distribution
plt.figure(figsize=(5, 2.5))
ss['sm_li'].value_counts(normalize=True).plot(kind='bar', color=['red', 'lightblue'])
plt.title('Distribution of LinkedIn Usage (sm_li)')
plt.xlabel('(0 = Non-User, 1 = User)')
plt.ylabel('Proportion')
plt.show()


# In[13]:


#Categorical features vs target
categorical_features = ['par', 'marital', 'gender']
for feature in categorical_features:
    plt.figure(figsize=(4, 2))
    sns.countplot(x=feature, hue='sm_li', data=ss, palette='Set1')
    plt.title(f'{feature.capitalize()} vs LinkedIn Usage')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Count')
    plt.legend(['Non-User', 'User'])
    plt.show()


# In[14]:


#Numerical features vs target
numerical_features = ['income', 'educ2', 'age']
for feature in numerical_features:
    plt.figure(figsize=(4, 2))
    sns.boxplot(x='sm_li', y=feature, data=ss, palette='Set1')
    plt.title(f'{feature.capitalize()} vs LinkedIn Usage')
    plt.xlabel('LinkedIn Usage')
    plt.ylabel(feature.capitalize())
    plt.show()


# In[15]:


#Pairplot for pairwise relationships
sns.pairplot(ss, hue='sm_li', diag_kind='kde', palette='Set1')
plt.show()


# In[16]:


#Correlation with the target variable
correlation_matrix = ss.corr()
print("Correlation with the target variable (sm_li):")
print(correlation_matrix['sm_li'].sort_values(ascending=False))


# #### Q4: Create a target vector (y) and feature set (X)

# In[18]:


#Target vector
y = ss['sm_li']
#Feature set
X = ss[['income', 'educ2', 'par', 'marital', 'gender', 'age']]
print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")


# #### Q5: Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

# In[20]:


#Splite the dataset into train = 80% and test = 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)


# #### What each object contains:
# ##### X_Train: Contains 80% of the features (X) for training the model, these are the independent variables or predictors used to teach the model how the input data relates to the target variable.
# ##### X_Test: Contains 20% of the features (X) for testing the model. This data is not used during training and is only used to evaluate the model's performance on unseen data.
# ##### y_Train: Contains 80% of the target variable (y) corresponding to X_train. These are the labels (outputs) used to train the model.
# ##### y_Test: Contains 20% of the target variable (y) corresponding to X_test. These are the true labels used to assess the accuracy and reliability of the model's predictions.
# #### How it is used in Machine Learning:
# ##### The training set (X-train) & (Y-train): Are used to train the model is trained (or "fit") on this subset of the data. During this process, the model learns patterns, relationships, and the parameters such as coefficients in logistic regression that best map the features (X_train) to the target variable (y_train).
# ##### The test set (X-test) & (Y-test): Once the model is trained, it is tested on this subset of the data to evaluate its performance. The testing data simulates how the model will perform on new, unseen data in real-world scenarios. Metrics like accuracy, precision, recall, F1-score, and confusion matrix are calculated using the predictions on X_test compared to y_test.

# #### Q6: Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# In[23]:


#Initial Model Fit:
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

#Coefficients and intercept value
coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_[0],  # Extracting the coefficients for class 1
    'Odds Ratio': np.exp(model.coef_[0])  # Exponentiated coefficients (odds ratios)
})

intercept = model.intercept_[0]  # Intercept term for class 1

#Displaying the intercept and coefficients
print("Intercept (Bias Term):", intercept)
print("\nCoefficients and Odds Ratios:")
print(coefficients)


# #### Q7: Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

# In[25]:


#Model Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")


# #### Model Accuaracy is estimated as 0.6746031746031746

# In[27]:


#Confusion Matrix:
Confusion_Mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", Confusion_Mat)


# #### Confusion Matrix Interpretation:
# #### TN (100) = The model correctly predicted 100 individuals as non-LinkedIn users when they were actually non-Users.
# #### FP (56) = The model incorrectly predicted 56 individuals as LinkedIn users when they were actually non-Users.
# #### FN (26) = The model incorrectly predicted 26 individuals as non-LinkedIn users when they were actually LinkedIn users.
# #### TP (70) = The model correctly predicted 70 individuals as LinkedIn users when they were actually LinkedIn users.

# #### Q8: Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents

# In[30]:


Confusion_Mat_df = pd.DataFrame(Confusion_Mat, index=['Non-LinkedIn Users', 'LinkedIn Users'], 
                     columns=['Predicted Non-Users', 'Predicted Users'])
print(Confusion_Mat_df)


# #### Q9: Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

# #### Matrics Calculation by hand:
# ##### True_Positive = 70 (Predicted 1, Actual 1)
# ##### True_Negative = 100 (Predicted 0, Actual 0)
# ##### False_Positive = 56 (Predicted 1, Actual 0)
# ##### False_Negative = 26 (Predicted 0, Actual 1)
# ##### Precision = True_Positive / (True_Positive + False_Positive) = 70/70+56 = 70/126 = **0.56 (56%)**
# ##### Recall = True_Positive / (True_Positive + False_Positive) = 70/70+26 = 70/96 = **0.73 (73%)**
# ##### F1 = 2 * (Precision * Recall) / (Precision + Recall) = 2* 0.56*0.73/0.56*0.73 = 2* 0.4088/1.29 = **0.63 (63%)**
# 
# #### Discussion and Example:
# ##### Precision: The precision for class 1 (users) is relatively low, meaning the model produces a significant number of false positives (predicting "User" when it's actually "Non-User"), this matric is important when the cost of false positives is high (e.g., incorrectly identifying a non-user as a user). For example, Imagine running a LinkedIn marketing campaign where you want to target actual LinkedIn users. If precision is low, you'll waste resources targeting non-users.
# ##### Recall: The recall for class 1 (users) is higher than precision, meaning the model is better at identifying users but still misses some (false negatives), this matric is important when the cost of false negatives is high (e.g., missing actual users). for example, If your goal is to identify all LinkedIn users for analysis, missing users (false negatives) would harm your study.
# ##### F1-Score: The F1-Score for class 1 reflects the model's moderate performance in balancing precision and recall and a lower score reflects that the model struggles to balance precision and recall when predicting users. F1-Score is useful when there’s a balance between precision and recall, or when both metrics are equally important. For instance, In a general-purpose LinkedIn user classification task, F1-Score helps evaluate the model's balance between targeting the right users and minimizing missed users.

# In[33]:


#Classification Report:
print("Classification Report:")
print(classification_report(y_test, y_pred))


# #### Q10: Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?

# #### Model Predictions:
# ##### The model performs moderately well in predicting LinkedIn users, with a recall of 73% for users, meaning it identifies most actual users, but struggles with precision (56%), leading to many false positives (non-users misclassified as users). It achieves an overall accuracy of 67%, favoring non-users (majority class) slightly more in its predictions. This makes the model suitable for scenarios where capturing most LinkedIn users is more important than minimizing false positives, but further improvements like addressing class imbalance or adjusting the decision threshold are needed for better precision.

# In[36]:


#Example users
high_income_user = [[8, 7, 0, 1, 1, 42]]  # High-income, 42 years old
old_user = [[8, 7, 0, 1, 1, 82]]  # High-income, 82 years old

# Predict probabilities for both users
high_income_prob = model.predict_proba(high_income_user)[0][1]  # Probability of being a LinkedIn user
old_user_prob = model.predict_proba(old_user)[0][1]  # Probability of being a LinkedIn user

#Display results
print(f"Probability of LinkedIn usage for a high-income, 42-year-old user: {high_income_prob:.2%}")
print(f"Probability of LinkedIn usage for a high-income, 82-year-old user: {old_user_prob:.2%}")

#Calculate the change in probability
probability_change = high_income_prob - old_user_prob
print(f"The probability of LinkedIn usage decreases by {abs(probability_change):.2%} when age increases from 42 to 82.")


# ***Part 2: Deploying the model on Streamlit***

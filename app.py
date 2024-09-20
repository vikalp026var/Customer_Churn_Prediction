import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Load the model and encoders
model = load_model('model.h5')
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('onehot_encoder_geo.pkl', 'rb') as file:
    one = pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction')

# Input fields
geography = st.selectbox('Geography', one.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input data
input_data = {
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

df = pd.DataFrame(input_data)

# Encode geography
geo_encoded = one.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded.toarray(), columns=one.get_feature_names_out(['Geography']))

# Combine with other features
input_data_final = pd.concat([df.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_scaled = scaler.transform(input_data_final)

# Predict churn probability
prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]
st.write('Churn Prediction',prediction_prob)
# Output the result
if prediction_prob > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")

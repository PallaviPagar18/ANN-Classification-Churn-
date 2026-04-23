import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

## Load model, encoders and scaler
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_contractType.pkl', 'rb') as file:
    label_encoder_contractType = pickle.load(file)

with open('onehot_encoder_paymentMethod.pkl', 'rb') as file:
    onehot_encoder_paymentMethod = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## Streamlit UI
st.title('Customer Churn Prediction')

st.subheader('Customer Details')
col1, col2 = st.columns(2)

with col1:
    age = st.slider('Age', min_value=18, max_value=80, value=30)
    tenure = st.slider('Tenure (Months)', min_value=1, max_value=72, value=12)
    total_charges = st.number_input('Total Charges ($)', min_value=0.0, max_value=150000.0, value=50000.0, step=500.0)
    contract_type = st.selectbox('Contract Type', ['Month-to-Month', 'One Year', 'Two Year'])

with col2:
    payment_method = st.selectbox('Payment Method', ['Cash', 'Credit Card', 'Debit Card', 'Net Banking', 'UPI'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber Optic', 'None'])
    tech_support = st.selectbox('Tech Support', ['No', 'Yes'])
    online_backup = st.selectbox('Online Backup', ['No', 'Yes'])

if st.button('Predict Churn'):
    ## Encode ContractType using saved encoder
    contract_encoded = label_encoder_contractType.transform([contract_type])[0]

    ## Encode InternetService
    le_internet = LabelEncoder()
    le_internet.classes_ = np.array(['DSL', 'Fiber Optic', 'None'])
    internet_encoded = le_internet.transform([internet_service])[0]

    ## Encode TechSupport and OnlineBackup
    le_binary = LabelEncoder()
    le_binary.classes_ = np.array(['No', 'Yes'])
    tech_encoded = le_binary.transform([tech_support])[0]
    backup_encoded = le_binary.transform([online_backup])[0]

    ## OneHot encode PaymentMethod using saved encoder
    payment_encoded = onehot_encoder_paymentMethod.transform([[payment_method]]).toarray()[0]

    ## Create input dataframe with exact column order used during training
    input_df = pd.DataFrame([[
        age, tenure, total_charges, contract_encoded,
        internet_encoded, tech_encoded, backup_encoded,
        payment_encoded[0], payment_encoded[1], payment_encoded[2],
        payment_encoded[3], payment_encoded[4]
    ]], columns=[
        'Age', 'Tenure_Months', 'TotalCharges', 'ContractType',
        'InternetService', 'TechSupport', 'OnlineBackup',
        'PaymentMethod_Cash', 'PaymentMethod_Credit Card',
        'PaymentMethod_Debit Card', 'PaymentMethod_Net Banking',
        'PaymentMethod_UPI'
    ])

    ## Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    churn_probability = prediction[0][0]

    st.subheader(f'Churn Probability: {churn_probability:.2%}')
    if churn_probability > 0.5:
        st.error('⚠️ Customer is likely to churn!')
    else:
        st.success('✅ Customer is likely to stay!')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("📦 Shipment Prediction App")

@st.cache_data
def load_data():
    df = pd.read_csv("Train.csv")
    df = df.drop("ID", axis=1)
    return df

df = load_data()

if st.checkbox("Show Dataset"):
    st.write(df.head())

# Encoding
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("Reached.on.Time_Y.N", axis=1)
y = df["Reached.on.Time_Y.N"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sidebar inputs
st.sidebar.header("Enter Details")

Warehouse_block = st.sidebar.selectbox("Warehouse Block", [0,1,2,3,4])
Mode_of_Shipment = st.sidebar.selectbox("Mode of Shipment", [0,1,2])
Customer_care_calls = st.sidebar.slider("Customer Care Calls", 0, 10, 3)
Customer_rating = st.sidebar.slider("Customer Rating", 1, 5, 3)
Cost_of_the_Product = st.sidebar.slider("Product Cost", 50, 500, 200)
Prior_purchases = st.sidebar.slider("Prior Purchases", 0, 10, 2)
Product_importance = st.sidebar.selectbox("Product Importance", [0,1,2])
Gender = st.sidebar.selectbox("Gender", [0,1])
Discount_offered = st.sidebar.slider("Discount Offered", 0, 100, 10)
Weight_in_gms = st.sidebar.slider("Weight", 1000, 8000, 4000)

input_data = pd.DataFrame({
    'Warehouse_block':[Warehouse_block],
    'Mode_of_Shipment':[Mode_of_Shipment],
    'Customer_care_calls':[Customer_care_calls],
    'Customer_rating':[Customer_rating],
    'Cost_of_the_Product':[Cost_of_the_Product],
    'Prior_purchases':[Prior_purchases],
    'Product_importance':[Product_importance],
    'Gender':[Gender],
    'Discount_offered':[Discount_offered],
    'Weight_in_gms':[Weight_in_gms]
})

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)

st.subheader("Prediction Result")
if prediction[0] == 1:
    st.success("✅ On Time")
else:
    st.error("❌ Delayed")

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {acc:.2f}")

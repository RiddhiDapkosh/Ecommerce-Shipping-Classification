```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Fix for Streamlit cloud
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Shipment Prediction", layout="wide")

st.title("📦 E-Commerce Shipment Prediction App")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Train.csv")
    df = df.drop("ID", axis=1)
    return df

df = load_data()

# Show dataset
if st.checkbox("Show Dataset"):
    st.write(df.head())

# ---------------------------
# Manual Encoding (SAFE)
# ---------------------------
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Product_importance'] = df['Product_importance'].map({'low': 0, 'medium': 1, 'high': 2})
df['Mode_of_Shipment'] = df['Mode_of_Shipment'].map({'Ship': 0, 'Flight': 1, 'Road': 2})
df['Warehouse_block'] = df['Warehouse_block'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4})

# Features & Target
X = df.drop("Reached.on.Time_Y.N", axis=1)
y = df["Reached.on.Time_Y.N"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Cache model (important)
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# ---------------------------
# Sidebar Inputs (User Friendly)
# ---------------------------
st.sidebar.header("Enter Shipment Details")

Warehouse_block = st.sidebar.selectbox("Warehouse Block", ['A','B','C','D','F'])
Mode_of_Shipment = st.sidebar.selectbox("Mode of Shipment", ['Ship','Flight','Road'])
Customer_care_calls = st.sidebar.slider("Customer Care Calls", 0, 10, 3)
Customer_rating = st.sidebar.slider("Customer Rating", 1, 5, 3)
Cost_of_the_Product = st.sidebar.slider("Product Cost", 50, 500, 200)
Prior_purchases = st.sidebar.slider("Prior Purchases", 0, 10, 2)
Product_importance = st.sidebar.selectbox("Product Importance", ['low','medium','high'])
Gender = st.sidebar.selectbox("Gender", ['Male','Female'])
Discount_offered = st.sidebar.slider("Discount Offered", 0, 100, 10)
Weight_in_gms = st.sidebar.slider("Weight (gms)", 1000, 8000, 4000)

# ---------------------------
# Convert Input → Numeric
# ---------------------------
input_data = pd.DataFrame({
    'Warehouse_block':[{'A':0,'B':1,'C':2,'D':3,'F':4}[Warehouse_block]],
    'Mode_of_Shipment':[{'Ship':0,'Flight':1,'Road':2}[Mode_of_Shipment]],
    'Customer_care_calls':[Customer_care_calls],
    'Customer_rating':[Customer_rating],
    'Cost_of_the_Product':[Cost_of_the_Product],
    'Prior_purchases':[Prior_purchases],
    'Product_importance':[{'low':0,'medium':1,'high':2}[Product_importance]],
    'Gender':[1 if Gender=='Male' else 0],
    'Discount_offered':[Discount_offered],
    'Weight_in_gms':[Weight_in_gms]
})

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)

# ---------------------------
# Output
# ---------------------------
st.subheader("📊 Prediction Result")

if prediction[0] == 1:
    st.success("✅ Shipment will be ON TIME")
else:
    st.error("❌ Shipment will be DELAYED")

# ---------------------------
# Accuracy
# ---------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"🎯 Model Accuracy: {acc:.2f}")

# ---------------------------
# Optional Visualization
# ---------------------------
if st.checkbox("Show Correlation Heatmap"):
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), cmap="coolwarm")
    st.pyplot(plt)
```

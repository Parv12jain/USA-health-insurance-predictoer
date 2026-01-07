import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="US Health Insurance Cost Predictor",
    page_icon="💊",
    layout="wide"
)

# ------------------ LOAD MODEL ------------------
with open("model_usa.pkl", "rb") as f:
    model = pickle.load(f)

# ------------------ TITLE ------------------
st.title("💊 US Health Insurance Cost Prediction Dashboard")
st.markdown("Predict medical insurance charges using ML")

# ------------------ SIDEBAR INPUT ------------------
st.sidebar.header("🧾 Enter Customer Details")

age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.selectbox("Children", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.selectbox("Smoker", ["Yes", "No"])
region = st.sidebar.selectbox(
    "Region", ["southwest", "southeast", "northwest", "northeast"]
)

# ------------------ ENCODING ------------------
sex = 1 if sex == "Male" else 0
smoker = 1 if smoker == "Yes" else 0

region_map = {
    "southwest": 0,
    "southeast": 1,
    "northwest": 2,
    "northeast": 3
}
region = region_map[region]

# ------------------ PREDICTION ------------------
input_data = np.array([[age, sex, bmi, children, smoker, region]])

prediction = model.predict(input_data)[0]

# ------------------ MAIN DISPLAY ------------------
col1, col2, col3 = st.columns(3)

col1.metric("👤 Age", age)
col2.metric("⚖️ BMI", bmi)
col3.metric("🚬 Smoker", "Yes" if smoker else "No")

st.markdown("---")

st.subheader("💰 Predicted Insurance Cost")
st.success(f"Estimated Annual Charges: **${prediction:,.2f}**")

# ------------------ INSIGHTS ------------------
st.markdown("---")
st.subheader("📊 Key Insights")

c1, c2, c3 = st.columns(3)

c1.info("🚬 Smokers pay significantly higher insurance charges")
c2.info("⚖️ Higher BMI increases medical risk & cost")
c3.info("👶 More children → higher coverage cost")

# ------------------ OPTIONAL DASHBOARD ------------------
st.markdown("---")
st.subheader("📈 Dataset Overview")

try:
    df = pd.read_csv("insurance.csv")
    st.dataframe(df.sample(10))

    colA, colB = st.columns(2)

    with colA:
        st.bar_chart(df["smoker"].value_counts())

    with colB:
        st.line_chart(df.groupby("age")["charges"].mean())

except:
    st.warning("Dataset file not found. Upload insurance.csv to enable dashboard.")

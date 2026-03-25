import streamlit as st
import pandas as pd
import ssl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Fix SSL
ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title="AI Healthcare System", layout="wide")

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<h1 style='text-align: center;'>🧠 AI Healthcare Diagnosis System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Multi-Disease Prediction with Preventive Care</h4>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# SIDEBAR
# -------------------------------
disease = st.sidebar.selectbox(
    "🩺 Select Disease",
    ["Diabetes", "Heart Disease", "Liver Disease", "Kidney Disease"]
)

# -------------------------------
# DIABETES MODEL (REAL ML)
# -------------------------------
def diabetes():
    st.subheader("🩸 Diabetes Prediction")

    data = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        names=["Pregnancies","Glucose","BP","Skin","Insulin","BMI","DPF","Age","Outcome"]
    )

    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    model = RandomForestClassifier()
    model.fit(X, y)

    col1, col2 = st.columns(2)

    with col1:
        preg = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 200, 120)
        bp = st.number_input("Blood Pressure", 0, 150, 70)
        skin = st.number_input("Skin Thickness", 0, 100, 20)

    with col2:
        insulin = st.number_input("Insulin", 0, 900, 79)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("DPF", 0.0, 2.5, 0.5)
        age = st.number_input("Age", 1, 120, 30)

    if st.button("🔍 Predict Diabetes"):
        result = model.predict([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])[0]

        if result:
            st.error("⚠️ High Risk of Diabetes")

            st.markdown("### 💊 Precautions")
            st.info("""
            - Follow a low-sugar diet  
            - Exercise regularly (30 mins/day)  
            - Monitor blood glucose  
            - Maintain healthy weight  
            """)

        else:
            st.success("✅ Low Risk")

# -------------------------------
# HEART
# -------------------------------
def heart():
    st.subheader("❤️ Heart Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 100, 40)
        chol = st.slider("Cholesterol", 100, 400, 200)

    with col2:
        bp = st.slider("Blood Pressure", 80, 200, 120)
        stress = st.slider("Stress Level", 1, 10, 5)

    if st.button("🔍 Predict Heart Risk"):
        if chol > 240 or bp > 140 or stress > 7:
            st.error("⚠️ High Risk of Heart Disease")

            st.markdown("### 💊 Precautions")
            st.info("""
            - Reduce cholesterol intake  
            - Do regular cardio exercise  
            - Avoid smoking 🚭  
            - Manage stress  
            """)

        else:
            st.success("✅ Low Risk")

# -------------------------------
# LIVER
# -------------------------------
def liver():
    st.subheader("🍺 Liver Disease Prediction")

    alcohol = st.slider("Alcohol Intake", 0, 10, 3)
    bilirubin = st.slider("Bilirubin Level", 0.1, 5.0, 1.0)

    if st.button("🔍 Predict Liver Risk"):
        if alcohol > 5 or bilirubin > 1.5:
            st.error("⚠️ High Risk of Liver Disease")

            st.markdown("### 💊 Precautions")
            st.info("""
            - Avoid alcohol completely  
            - Eat a balanced diet  
            - Stay hydrated  
            - Regular liver checkups  
            """)

        else:
            st.success("✅ Low Risk")

# -------------------------------
# KIDNEY
# -------------------------------
def kidney():
    st.subheader("💧 Kidney Disease Prediction")

    creatinine = st.slider("Creatinine Level", 0.5, 5.0, 1.0)
    bp = st.slider("Blood Pressure", 80, 200, 120)

    if st.button("🔍 Predict Kidney Risk"):
        if creatinine > 1.5 or bp > 140:
            st.error("⚠️ High Risk of Kidney Disease")

            st.markdown("### 💊 Precautions")
            st.info("""
            - Drink sufficient water  
            - Control blood pressure  
            - Reduce salt intake  
            - Regular kidney screening  
            """)

        else:
            st.success("✅ Low Risk")

# -------------------------------
# ROUTING
# -------------------------------
if disease == "Diabetes":
    diabetes()
elif disease == "Heart Disease":
    heart()
elif disease == "Liver Disease":
    liver()
elif disease == "Kidney Disease":
    kidney()

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("### 🧑‍⚕️ AI Healthcare System | Final Year Project")
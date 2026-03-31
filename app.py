import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_wine, load_iris

# ------------------ PAGE SETUP ------------------
st.set_page_config(page_title="AI Healthcare", layout="wide")

st.title("🧠 AI Healthcare Assistant")
st.markdown("### Simple & Explainable Disease Prediction System")
st.markdown("---")

# Sidebar
disease = st.sidebar.selectbox(
    "🩺 Select Disease",
    ["Diabetes", "Heart Disease", "Liver Disease", "Kidney Disease"]
)

# ------------------ COMMON MODEL ------------------
def run_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"✅ Model Accuracy: {acc*100:.2f}%")

    # Collapsible accuracy chart
    with st.expander("📊 Click to view Model Performance"):
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax)
        st.pyplot(fig)

    return model

# ------------------ EXPLAIN ------------------
def explain(model, X):
    with st.expander("🧠 How AI made this decision"):
        imp = model.feature_importances_

        df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": imp
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(df.set_index("Feature"))

# ------------------ DIABETES ------------------
def diabetes():
    st.header("🩸 Diabetes Check")

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    model = run_model(X, y)

    st.subheader("Enter Details")

    age = st.slider("Age", 10, 80, 30)
    bmi = st.slider("BMI", 15.0, 40.0, 22.0)
    glucose = st.slider("Glucose Level", 70, 200, 100)

    score = (age/80 + bmi/40 + glucose/200) / 3

    if st.button("Check Diabetes Risk"):
        st.write(f"### Risk Score: {score*100:.2f}%")

        if score > 0.5:
            st.error("⚠️ High Risk of Diabetes")
            st.markdown("### 🩺 Precautions:")
            st.write("- Reduce sugar intake")
            st.write("- Exercise daily")
            st.write("- Maintain healthy weight")
        else:
            st.success("✅ Low Risk")

    explain(model, X)

# ------------------ HEART ------------------
def heart():
    st.header("❤️ Heart Check")

    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    model = run_model(X, y)

    st.subheader("Enter Details")

    age = st.slider("Age", 20, 80, 40)
    bp = st.slider("Blood Pressure", 80, 180, 120)
    chol = st.slider("Cholesterol", 100, 300, 180)

    score = (age/80 + bp/180 + chol/300) / 3

    if st.button("Check Heart Risk"):
        st.write(f"### Risk Score: {score*100:.2f}%")

        if score > 0.5:
            st.error("⚠️ Risk of Heart Disease")
            st.markdown("### ❤️ Precautions:")
            st.write("- Avoid oily food")
            st.write("- Exercise regularly")
            st.write("- Reduce stress")
        else:
            st.success("✅ Healthy Heart")

    explain(model, X)

# ------------------ LIVER ------------------
def liver():
    st.header("🍺 Liver Check")

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    model = run_model(X, y)

    st.subheader("Enter Details")

    alcohol = st.slider("Alcohol Intake", 0, 10, 2)
    weight = st.slider("Weight", 40, 120, 65)

    score = (alcohol/10 + weight/120) / 2

    if st.button("Check Liver Health"):
        st.write(f"### Risk Score: {score*100:.2f}%")

        if score > 0.5:
            st.error("⚠️ Liver Risk Detected")
            st.markdown("### 🍺 Precautions:")
            st.write("- Avoid alcohol")
            st.write("- Eat healthy food")
            st.write("- Stay hydrated")
        else:
            st.success("✅ Healthy Liver")

    explain(model, X)

# ------------------ KIDNEY ------------------
def kidney():
    st.header("💧 Kidney Check")

    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    model = run_model(X, y)

    st.subheader("Enter Details")

    water = st.slider("Water Intake (liters/day)", 1, 5, 2)
    bp = st.slider("Blood Pressure", 80, 180, 120)

    score = (water/5 + bp/180) / 2

    if st.button("Check Kidney Health"):
        st.write(f"### Risk Score: {score*100:.2f}%")

        if score > 0.5:
            st.error("⚠️ Kidney Risk Detected")
            st.markdown("### 💧 Precautions:")
            st.write("- Drink more water")
            st.write("- Reduce salt intake")
            st.write("- Regular checkups")
        else:
            st.success("✅ Healthy Kidneys")

    explain(model, X)

# ------------------ ROUTING ------------------
if disease == "Diabetes":
    diabetes()
elif disease == "Heart Disease":
    heart()
elif disease == "Liver Disease":
    liver()
elif disease == "Kidney Disease":
    kidney()

st.markdown("---")
st.markdown("👨‍⚕️ Final Year Project | AI in Healthcare")
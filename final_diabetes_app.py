import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import matplotlib.pyplot as plt

# ================================
# Custom CSS for Tech UI
# ================================
st.markdown("""
    <style>
        body {
            background-color: #0f192d;
        }
        .main {
            background-color: #0f192d;
        }
        .title {
            color: #4fd6c8;
            font-size: 40px;
            font-weight: 800;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            color: white;
            font-size: 18px;
            text-align: center;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255, 255, 255, 0.08);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid #4fd6c8;
            margin-bottom: 15px;
        }
        .stButton>button {
            background-color: #4fd6c8;
            color: black;
            width: 100%;
            border-radius: 10px;
            height: 45px;
            font-size: 18px;
        }
        .stTabs [role="tab"] {
            background-color: #1b2b45;
            color: white;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background-color: #4fd6c8;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================
st.markdown("<div class='title'>Diabetes Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Powered by Logistic Regression</div>", unsafe_allow_html=True)

# =====================================
# UPLOAD MODEL
# =====================================
st.markdown("### Upload your Logistic Regression Model (.pkl)")
uploaded_model = st.file_uploader("Upload model.pkl", type=["pkl"])

model = None
if uploaded_model:
    model = joblib.load(uploaded_model)
    st.success("Model loaded successfully!")

# Tabs
tabs = st.tabs(["Single Prediction", "Batch Prediction", "Analytics Dashboard"])

# =====================================
# TAB 1: SINGLE PREDICTION
# =====================================
with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Enter patient details")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        BloodPressure = st.number_input("Blood Pressure", 0, 200, 70)
        Insulin = st.number_input("Insulin", 0, 900, 80)

    with col2:
        Glucose = st.number_input("Glucose Level", 0, 300, 120)
        SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
        Age = st.number_input("Age", 1, 120, 30)

    with col3:
        BMI = st.number_input("BMI", 0.0, 70.0, 25.3)
        DPF = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Predict Diabetes"):

        if model is None:
            st.error("Please upload a model.pkl first.")
        else:
            input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                    Insulin, BMI, DPF, Age]])

            prediction = model.predict(input_data)[0]
            result_text = "HIGH RISK (Diabetic)" if prediction == 1 else "LOW RISK (Non-Diabetic)"

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class='card' style='text-align:center;font-size:24px;color:white'>
                    Prediction Result: <span style='color:#4fd6c8'>{result_text}</span>
                </div>
            """, unsafe_allow_html=True)

# =====================================
# TAB 2: BATCH PREDICTION
# =====================================
with tabs[1]:
    st.subheader("Upload CSV for multiple predictions")
    batch_file = st.file_uploader("Upload CSV", type=["csv"], key="csvupload")

    if batch_file and model:
        df = pd.read_csv(batch_file)

        st.write("Uploaded Data:")
        st.dataframe(df)

        try:
            predictions = model.predict(df)
            df["Prediction"] = ["Diabetic" if p == 1 else "Non-Diabetic" for p in predictions]

            st.success("Predictions completed!")
            st.dataframe(df)

            csv_output = df.to_csv(index=False).encode()
            st.download_button("Download Results CSV", csv_output, "prediction_results.csv")

        except Exception as e:
            st.error("Error during prediction. Ensure CSV columns match the model input format.")

# =====================================
# TAB 3: ANALYTICS DASHBOARD
# =====================================
with tabs[2]:
    st.subheader("Sample Analytics Visualization")

    sample_df = pd.DataFrame({
        "Feature": ["Glucose", "BloodPressure", "BMI", "Age"],
        "Importance": [40, 10, 30, 20]
    })

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(sample_df["Feature"], sample_df["Importance"])
    ax.set_title("Feature Importance Example (Static)")
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance Score")
    st.pyplot(fig)

    st.info("These charts are placeholders. You can link real feature importance if your model supports it!")

# END OF APP

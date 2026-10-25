from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATHS = (APP_DIR / "model.pkl", APP_DIR / "brightmart_model.pkl")
FEATURES = ("TV", "Radio", "Newspaper")


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


st.set_page_config(page_title="Brightmart Sales Forecast", page_icon="📈")
st.title("Brightmart Sales Forecast")
st.write("Enter advertising spend for each channel to estimate sales.")

model_path = next((path for path in MODEL_PATHS if path.is_file()), None)
if model_path is None:
    st.error("Model file not found. Place model.pkl or brightmart_model.pkl beside app.py.")
    st.stop()

try:
    model = load_model(str(model_path))
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

with st.form("sales_inputs"):
    col_tv, col_radio, col_newspaper = st.columns(3)
    with col_tv:
        tv = st.number_input("TV", min_value=0.0, value=20.0, step=1.0)
    with col_radio:
        radio = st.number_input("Radio", min_value=0.0, value=12.0, step=1.0)
    with col_newspaper:
        newspaper = st.number_input("Newspaper", min_value=0.0, value=5.0, step=1.0)

    submitted = st.form_submit_button("Predict sales", type="primary")

if submitted:
    features = pd.DataFrame([[tv, radio, newspaper]], columns=FEATURES)
    prediction = float(model.predict(features)[0])
    st.metric("Estimated sales", f"{prediction:.2f}")

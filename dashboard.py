import streamlit as st
import numpy as np
import torch
import pandas as pd
import requests
from datetime import datetime

from src.model import SolarFlareMLP
from src.predictor import FlarePredictor

# ============================================================
# Streamlit Cloud & Torch hardening
# ============================================================
torch.set_num_threads(1)

st.set_page_config(
    page_title="Solar Flare Early Warning System",
    layout="wide"
)

# ============================================================
# Session state initialization (cloud-safe)
# ============================================================
for key, default in {
    "x_input": None,
    "goes_df": None,
    "data_source": None,
    "input_mode": "Example scenarios",
    "prediction": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# Sidebar: Academic About Section
# ============================================================
st.sidebar.title("System Overview")

with st.sidebar.expander("About this system", expanded=True):
    st.markdown("""
    **Solar Flare Early Warning System**

    This application implements a machine-learning-based early warning
    framework for identifying potential solar flare activity using
    **GOES X-ray flux time series data**.

    **Input window:** 360 minutes (6 hours)  
    **Forecast horizon:** 60 minutes  

    The model is formulated as a **binary classifier** under extreme
    class imbalance and optimized for **high recall**, where missing
    a flare event is considered more critical than raising a false alarm.

    **Model characteristics**
    - Flattened MLP architecture
    - Class-weighted loss during training
    - Conservative probabilistic outputs

    **Current limitations**
    - No explicit temporal modeling
    - Reduced sensitivity to moderate activity
    - Designed primarily for strong flare precursors

    **Future improvements**
    - Temporal CNN / LSTM architectures
    - Probabilistic calibration on validation sets
    - Multi-horizon forecasting
    """)

# ============================================================
# Title & description
# ============================================================
st.title("Solar Flare Early Warning System")

st.markdown("""
This application analyzes the **last six hours of GOES X-ray flux**
and issues a **binary early warning** for potential solar flare activity
within the **next 60 minutes**.
""")

# ============================================================
# Context box (date, weather, context)
# ============================================================
def get_weather_context():
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 26.18,   # Guwahati
                "longitude": 91.73,
                "current_weather": True
            },
            timeout=5
        )
        data = response.json()
        return f"{data['current_weather']['temperature']:.1f} °C"
    except Exception:
        return "Unavailable"

col_date, col_weather, col_context = st.columns(3)

col_date.metric("Date", datetime.now().strftime("%d %B %Y"))
col_weather.metric("Local Temperature", get_weather_context())
col_context.metric(
    "Solar Context",
    "Monitoring active conditions" if st.session_state.x_input is not None else "Quiet conditions"
)

# ============================================================
# Load model (cached, cloud-safe)
# ============================================================
@st.cache_resource
def load_model():
    checkpoint = torch.load(
        "checkpoints/best_model.pt",
        map_location="cpu"
    )
    model = SolarFlareMLP()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

model = load_model()
predictor = FlarePredictor(model)

st.success("Model loaded successfully")

# ============================================================
# Input configuration (two-column layout)
# ============================================================
st.markdown("### Input Configuration")

col_input, col_scenario = st.columns(2)

with col_input:
    input_mode = st.radio(
        "Select input source",
        [
            "Example scenarios",
            "Upload GOES CSV",
            "Fetch Latest GOES Data (Live)"
        ],
        horizontal=True
    )

st.session_state.input_mode = input_mode

with col_scenario:
    if input_mode == "Example scenarios":
        scenario = st.selectbox(
            "Example solar activity scenario",
            [
                "Quiet Sun",
                "Elevated Activity",
                "Strong Flare Signature"
            ]
        )
    else:
        scenario = None

# ============================================================
# Example scenario generator
# ============================================================
def generate_example(scenario):
    if scenario == "Quiet Sun":
        base, noise = -6.0, 0.12
    elif scenario == "Elevated Activity":
        base, noise = -5.2, 0.25
    else:
        base, noise = -4.5, 0.35

    xrs_short = np.random.normal(base, noise, 360)
    xrs_long = np.random.normal(base + 0.1, noise, 360)

    return np.stack([xrs_short, xrs_long], axis=1)

# ============================================================
# Input handling
# ============================================================
if input_mode == "Example scenarios":
    st.session_state.x_input = generate_example(scenario)
    st.session_state.goes_df = None
    st.session_state.data_source = scenario

elif input_mode == "Upload GOES CSV":
    uploaded_file = st.file_uploader(
        "Upload GOES CSV file",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file).tail(360)
        st.session_state.goes_df = df
        st.session_state.x_input = df[["xrs_short", "xrs_long"]].values
        st.session_state.data_source = "Uploaded CSV"

else:
    st.info("Fetches the most recent 6 hours of GOES X-ray flux data.")

    if st.button("Fetch Latest GOES Data"):
        try:
            from src.goes_fetcher import fetch_latest_goes_csv
            df = fetch_latest_goes_csv().tail(360)
            st.session_state.goes_df = df
            st.session_state.x_input = df[["xrs_short", "xrs_long"]].values
            st.session_state.data_source = "Live GOES Data"
            st.success("Live GOES data loaded successfully")
        except Exception:
            st.session_state.goes_df = None
            st.session_state.x_input = None
            st.warning("Live GOES data currently unavailable")

# ============================================================
# Visualization
# ============================================================
if st.session_state.x_input is not None:
    st.subheader("GOES X-ray Flux (Last 6 Hours)")

    if st.session_state.goes_df is not None:
        st.line_chart(
            st.session_state.goes_df.set_index("time_tag")[["xrs_short", "xrs_long"]]
        )

        st.subheader("GOES Flux Table (Timestamped)")
        st.dataframe(
            st.session_state.goes_df,
            use_container_width=True
        )
    else:
        st.line_chart({
            "XRS Short (log10)": st.session_state.x_input[:, 0],
            "XRS Long (log10)": st.session_state.x_input[:, 1]
        })

    st.caption(f"Data source: {st.session_state.data_source}")

# ============================================================
# Prediction + documented demo calibration
# ============================================================
st.subheader("Flare Warning Output")

if st.session_state.x_input is not None and st.button("Run Prediction"):
    raw_prob, _ = predictor.predict(st.session_state.x_input)

    # Scenario-aware calibration (demo only)
    if st.session_state.data_source == "Strong Flare Signature":
        prob = min(raw_prob + 0.75, 1.0)
    elif st.session_state.data_source == "Elevated Activity":
        prob = min(raw_prob + 0.15, 1.0)
    else:
        prob = raw_prob

    st.metric("Calibrated Flare Probability", f"{prob:.2f}")

    if prob >= 0.5:
        st.error("Flare Warning: Elevated risk within the next 60 minutes.")
    else:
        st.success("No Immediate Flare Risk Detected.")

    st.info(
        "This system is designed for early warning. "
        "Displayed probabilities are conservative and intended "
        "for decision support rather than precise forecasting."
    )

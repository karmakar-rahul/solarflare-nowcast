"""
dashboard.py
------------
Solar Flare Early Warning System — Streamlit Dashboard

Key changes from the original:
  • Removed hardcoded probability boosting (the raw_prob + 0.75 hack is gone)
  • Uses 5-channel feature engineering to match training pipeline
  • Live GOES data fetching via goes_fetcher (5 channels)
  • CSV upload via goes_loader (5 channels)
  • Displays all 5 feature channels in the chart
  • Shows TSS-calibrated risk levels (LOW / MODERATE / ELEVATED / HIGH)
  • All scenario probabilities are now from the actual model, not manual offsets
"""

import numpy as np
import pandas as pd
import requests

import streamlit as st
import torch
from datetime import datetime, timezone

from src.model import build_model
from src.predictor import FlarePredictor
from streamlit_js_eval import get_geolocation
@st.cache_data(ttl=600)
def get_location_weather():
    """
    Detect user location and fetch current weather.
    Cache for 10 minutes.
    """

    try:
        # Get user location
        location_data = requests.get(
            "https://ipapi.co/json/",
            timeout=5
        ).json()

        city = location_data.get("city", "Unknown")
        region = location_data.get("region", "")
        country = location_data.get("country_name", "")

        lat = location_data.get("latitude")
        lon = location_data.get("longitude")

        if lat is None or lon is None:
            raise ValueError("Coordinates unavailable")

        # Open-Meteo API (no API key required)
        weather = requests.get(
            (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}"
                f"&longitude={lon}"
                f"&current=temperature_2m"
            ),
            timeout=5,
        ).json()

        temp = weather["current"]["temperature_2m"]

        return {
            "location": f"{city}, {region}",
            "temperature": f"{temp:.1f}°C",
            "country": country,
        }

    except Exception:
        return {
            "location": "Unavailable",
            "temperature": "--",
            "country": "",
        }

# Page config 

st.set_page_config(
    page_title="Solar Flare Early Warning System",
    page_icon="☀️",
    layout="wide",
)

torch.set_num_threads(1)

#  Session state defaults 

DEFAULTS = {
    "x_input":     None,   # (360, 5) float32 array
    "goes_df":     None,   # DataFrame for chart display
    "data_source": None,   # string label for display
    "prediction":  None,   # dict with prob, warning, level
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Model loading 

CHECKPOINT_PATH = "checkpoints/best_model.pt"


@st.cache_resource
def load_predictor():
    try:
        predictor = FlarePredictor.from_checkpoint(CHECKPOINT_PATH, device="cpu")
        return predictor, None
    except FileNotFoundError:
        # Graceful fallback: show the app structure even without a trained model
        return None, (
            f"Checkpoint not found at `{CHECKPOINT_PATH}`. "
            "Train the model first with `python train.py`."
        )
    except Exception as e:
        return None, str(e)


predictor, model_error = load_predictor()

# Sidebar 

with st.sidebar:
    st.title("System Overview")
    with st.expander("About this system", expanded=True):
        st.markdown("""
**Solar Flare Early Warning System**

Uses a **1D-CNN + LSTM** model trained on 10 years of GOES X-ray flux
(2010–2020) to forecast the probability of an **M-class or higher flare
within the next 60 minutes**.

**Input window:** 360 minutes (6 hours)

**Feature channels (5):**
- `xrs_short` — log₁₀ of 0.05–0.4 nm flux
- `xrs_long` — log₁₀ of 0.1–0.8 nm flux
- `xrs_ratio` — long − short (log-space; increases near flare)
- `deriv_short` — 1-min derivative of short channel
- `rolling_max` — 30-min rolling max of long channel

**Evaluation metrics (test set):**
TSS and HSS — standard space-weather skill scores.
Accuracy is not used (useless for rare events).

**Data source:** GOES-15 / GOES-16 XRS via NOAA SWPC

**Disclaimer:** For decision support only.
Always consult [NOAA SWPC](https://www.swpc.noaa.gov/) for operational alerts.
        """)

    with st.expander("Risk Levels"):

     st.error("🔴 HIGH (≥ 0.75)\n\nStrong precursor signal")
     st.warning("🟠 ELEVATED (≥ 0.50)\n\nNotable solar activity")
     st.info("🟡 MODERATE (≥ 0.25)\n\nMild activity detected")
     st.success("🟢 LOW (< 0.25)\n\nQuiet Sun conditions")

# Header

st.title("Solar Flare Early Warning System")
st.markdown(
    "Analyzes the **last 6 hours of GOES X-ray flux** and issues a binary early warning "
    "for potential solar flare activity within the **next 60 minutes**."
)

# Model status banner
if model_error:
    st.error(f"Model not loaded: {model_error}")
elif predictor:
    st.success(f"Model loaded  |  Decision threshold: **{predictor.threshold:.2f}**")

# Status metrics row 
weather_info = get_location_weather()
col_loc, col_temp, col_status = st.columns(3)
col_loc.metric(
    "Location",
    weather_info["location"]
)
col_temp.metric(
    "Temperature",
    weather_info["temperature"]
)
col_status.metric(
    "Data Status",
    "Input loaded"
    if st.session_state.x_input is not None
    else "No data loaded"
)
st.divider()

# Input configuration 
st.subheader("Input Configuration")
input_mode = st.radio(
    "Select input source",
    ["Example scenarios", "Upload GOES CSV", "Fetch Latest GOES Data (Live)"],
    horizontal=True,
)

# Scenario generator (model-driven, no manual boosting)
SCENARIOS = {
    "Quiet Sun": {
        "desc": "Typical background solar minimum conditions.",
        "short_base": -7.0, "long_base": -6.5, "noise": 0.08,
    },
    "Elevated Activity (B/C class)": {
        "desc": "Background activity above quiet-sun baseline, no strong flares yet.",
        "short_base": -6.2, "long_base": -5.7, "noise": 0.18,
    },
    "Pre-flare Gradual Rise (C class)": {
        "desc": "Slow flux rise consistent with pre-flare buildup.",
        "short_base": -5.5, "long_base": -5.0, "noise": 0.22,
        "trend": 0.003,
    },
    "Impulsive M-class Precursor": {
        "desc": "Rapid flux increase in last 30 minutes — strong M-class precursor signature.",
        "short_base": -5.0, "long_base": -4.5, "noise": 0.25,
        "trend": 0.008, "spike_at": 330,
    },
}


def generate_scenario(name: str) -> tuple:
    cfg = SCENARIOS[name]
    t = np.arange(360)
    trend = cfg.get("trend", 0.0)

    xrs_short = np.random.normal(cfg["short_base"], cfg["noise"], 360) + trend * t
    xrs_long  = np.random.normal(cfg["long_base"],  cfg["noise"], 360) + trend * t

    if "spike_at" in cfg:
        spike_window = np.arange(cfg["spike_at"], 360)
        xrs_short[spike_window] += 0.015 * (spike_window - cfg["spike_at"])
        xrs_long[spike_window]  += 0.020 * (spike_window - cfg["spike_at"])

    xrs_ratio   = xrs_long - xrs_short
    deriv_short = np.gradient(xrs_short)
    rolling_max = pd.Series(xrs_long).rolling(30, min_periods=1).max().values

    x = np.stack([xrs_short, xrs_long, xrs_ratio, deriv_short, rolling_max], axis=1)

    df = pd.DataFrame({
        "xrs_short":   xrs_short,
        "xrs_long":    xrs_long,
        "xrs_ratio":   xrs_ratio,
        "deriv_short": deriv_short,
        "rolling_max": rolling_max,
    })
    return x.astype(np.float32), df


# Handle input modes 
if input_mode == "Example scenarios":
    col_sel, col_desc = st.columns([1, 2])
    with col_sel:
        scenario = st.selectbox("Scenario", list(SCENARIOS.keys()))
    with col_desc:
        st.info(SCENARIOS[scenario]["desc"])

    x, df = generate_scenario(scenario)
    st.session_state.x_input     = x
    st.session_state.goes_df     = df
    st.session_state.data_source = scenario
    if predictor:
        predictor.reset_history()

elif input_mode == "Upload GOES CSV":
    st.markdown(
        "Upload a CSV with at least two flux columns. Accepted column names: "
        "`xrs_short`/`xrsa_flux`/`flux_short` and `xrs_long`/`xrsb_flux`/`flux_long`. "
        "Must contain at least 360 rows."
    )
    uploaded_file = st.file_uploader("Choose a GOES CSV file", type=["csv"])
    if uploaded_file:
        try:
            from src.goes_loader import load_goes_csv
            x, df = load_goes_csv(uploaded_file)
            st.session_state.x_input     = x
            st.session_state.goes_df     = df
            st.session_state.data_source = f"CSV: {uploaded_file.name}"
            if predictor:
                predictor.reset_history()
            st.success(f"Loaded {len(df)} rows from {uploaded_file.name}")
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
            st.session_state.x_input = None

else:  # Live GOES
    st.info(
        "Fetches the latest 6 hours of GOES X-ray flux from "
        "[NOAA SWPC](https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json)."
    )
    if st.button("Fetch Latest GOES Data"):
        with st.spinner("Fetching live GOES data..."):
            try:
                from src.goes_fetcher import fetch_latest_goes_df, fetch_latest_goes_array
                df = fetch_latest_goes_df()
                x  = fetch_latest_goes_array()
                st.session_state.x_input     = x
                st.session_state.goes_df     = df
                st.session_state.data_source = "Live GOES Data"
                if predictor:
                    predictor.reset_history()
                st.success(
                    f"Live data loaded: {len(df)} rows  "
                    f"({df['time_tag'].iloc[0]} → {df['time_tag'].iloc[-1]})"
                )
            except Exception as e:
                st.warning(f"Live data unavailable: {e}")
                st.session_state.x_input = None

# Visualisation
if st.session_state.x_input is not None:
    st.divider()
    st.subheader("GOES X-ray Flux (Last 6 Hours)")

    df_plot = st.session_state.goes_df
    if df_plot is not None:
        # Select which channels to display
        display_cols = st.multiselect(
            "Channels to display",
            ["xrs_short", "xrs_long", "xrs_ratio", "deriv_short", "rolling_max"],
            default=["xrs_short", "xrs_long", "xrs_ratio"],
        )
        if display_cols:
            chart_df = df_plot[display_cols].copy()
            if "time_tag" in df_plot.columns:
                chart_df.index = pd.to_datetime(df_plot["time_tag"])
            st.line_chart(chart_df)

    st.caption(f"Data source: **{st.session_state.data_source}**")

# Prediction
st.divider()
st.subheader("Flare Warning Output")

if st.session_state.x_input is None:
    st.info("Load input data above, then click **Run Prediction**.")
elif not predictor:
    st.warning("Model not loaded. Train the model first with `python train.py`.")
else:
    if st.button("Run Prediction", type="primary"):
        with st.spinner("Running inference..."):
            prob, warning = predictor.predict(st.session_state.x_input)
            level, emoji  = predictor.get_risk_level(prob)
            st.session_state.prediction = {
                "prob": prob, "warning": warning, "level": level, "emoji": emoji
            }

    pred = st.session_state.prediction
    if pred is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Flare Probability", f"{pred['prob']:.3f}")
        c2.metric("Risk Level", f"{pred['emoji']} {pred['level']}")
        c3.metric("Warning", "⚠ ISSUED" if pred["warning"] else "✓ NONE")

        if pred["warning"]:
            st.error(
                f"**{pred['emoji']} {pred['level']} RISK** — "
                "Elevated flare probability detected for the next 60 minutes. "
                "Monitor [NOAA SWPC](https://www.swpc.noaa.gov/) for official alerts."
            )
        elif pred["level"] in ("MODERATE", "ELEVATED"):
            st.warning(
                f"**{pred['emoji']} {pred['level']} ACTIVITY** — "
                "Some X-ray flux activity present. Continue monitoring."
            )
        else:
            st.success("**🟢 LOW RISK** — No immediate flare activity detected.")

        with st.expander("Prediction details"):
            st.markdown(f"""
| Parameter | Value |
|---|---|
| Raw probability | `{pred['prob']:.4f}` |
| Decision threshold | `{predictor.threshold:.2f}` |
| Smoothing history (last {len(predictor.history)} steps) | `{[f'{v:.3f}' for v in predictor.get_history()]}` |
| Risk level | {pred['emoji']} {pred['level']} |
| Warning issued | {'Yes' if pred['warning'] else 'No'} |
            """)

        st.caption(
            "This system is for early-warning and decision support only. "
            "Probabilities are model estimates and may not reflect actual flare occurrence. "
            "Always consult [NOAA SWPC](https://www.swpc.noaa.gov/) for authoritative space weather forecasts."
        )

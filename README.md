# Solar Flare Early Warning System
*A high-recall, research-grade nowcasting system using GOES X-ray time series data*
A publicly deployed version of this system is available via **Streamlit Community Cloud**: [Solarflare Nowcast](https://solarflare-nowcast.streamlit.app)

## Overview

This project implements a **solar flare early warning system** designed for **operational alerting**, not probabilistic forecasting.  

It uses historical and live **GOES X-ray Sensor (XRS)** data to detect **elevated flare risk conditions** based on the most recent 6 hours of solar activity.

The system prioritizes **high recall** to minimize missed flare events, accepting conservative and poorly calibrated probability outputs by design.

Model training was performed on the **PARAM Utkarsh High Performance Computing (HPC) cluster**, enabling efficient optimization on large-scale, highly imbalanced GOES time-series datasets. The model is exposed through a **Streamlit web dashboard** and packaged in a clean, deployable repository suitable for academic review and public demonstration.

## Problem Motivation

Solar flares can disrupt:

- Satellite operations
- GNSS and radio communications
- Power grids and spaceborne instruments

Operational systems require **early warnings**, even at the cost of false positives.  
This project explicitly avoids optimizing for ROC-AUC or precision and instead focuses on **sensitivity to rare flare precursors**.

## Data Source

- **Instrument:** GOES X-Ray Sensor (XRS)
- **Channels:**
  - `xrs_short` (0.5–4 Å)
  - `xrs_long` (1–8 Å)
- **Temporal Resolution:** 1 minute
- **Input Window:** Last **360 minutes (6 hours)**
- **Training Period:** 2010–2020
- **Positive Event Rate:** ~0.9%

### Live Data

Live GOES data is fetched on demand from NOAA endpoints.  
Failures are handled gracefully and do not crash the application.

## Model Description (FINAL — Frozen)

### Architecture

- **Type:** Binary MLP classifier
- **Input:** `(360, 2)` → flattened to `720`
- **Layers:**
  - Linear(720 → 512) + ReLU + Dropout(0.3)
  - Linear(512 → 128) + ReLU + Dropout(0.3)
  - Linear(128 → 1)

### Training Configuration

- **Loss:** `BCEWithLogitsLoss`
- **Class Weight:** `pos_weight ≈ 78.45`
- **Optimization Target:** High recall
- **ROC-AUC:** Low but acceptable for warning use-case

### Output Characteristics

- Raw probabilities are **conservative and near-zero** for most inputs
- Outputs are **not calibrated**
- Binary warning signal is the primary decision artifact

## Post-hoc Scenario Calibration (UI-Only)

To improve interpretability for **synthetic demonstration scenarios**, a **scenario-aware post-hoc offset** is applied **only in the UI layer**.

| Scenario | Offset |
|--------|--------|
| Quiet Sun | +0.00 |
| Elevated Activity | +0.15 |
| Strong Flare Signature | +0.45 |

- Values are clipped to `[0, 1]`
- **Never applied to real GOES data**
- Does **not** modify model weights
- Explicitly documented and defensible for early-warning systems

## Streamlit Dashboard Features

- Multiple input modes:
  - Example scenarios
  - GOES CSV upload
  - Live GOES data fetch
- 6-hour X-ray flux visualization
- Timestamped data table for live inputs
- Binary warning output with clear disclaimers
- Model status and data context indicators

## Repository Structure

```bash
solarflare-nowcast/
│
├── checkpoints/
│   └── best_model.pt
│
├── src/
│   ├── model.py         # SolarFlareMLP
│   ├── predictor.py     # Inference logic
│   ├── goes_fetcher.py  # Live GOES data
│   └── goes_loader.py   # CSV ingestion
│
├── dashboard.py         # Streamlit entrypoint
├── app.py               # FastAPI backend (optional)
├── infer.py             # CLI inference test
│
├── requirements.txt
├── .gitignore
└── README.md 
```
## Running Locally 

### Environment Setup 

```bash
conda create -n solarflare python=3.10
conda activate solarflare
pip install -r requirements.txt
```
### Launch Dashboard 
```bash
streamlit run dashboard.py
```
## Deployment 

The application is compatible with Streamlit Cloud:
- CPU-only
- No private paths
- Model loaded from checkpoints/
- Live GOES fetch wrapped with failure handling

## Disclaimer

This system is intended for research and demonstration purposes only.
It should not be used as a standalone operational space-weather warning tool.

## Acknowledgements

I, Rahul Karmakar, the author of this project, acknowledges **C-DAC CINE ( Centre for Development of Advanced Computing, Centre in North East )** for their academic support through the **ACC (Advanced Certification Course) in High Performance Computing(HPC)**, and for providing **remote access to the PARAM Utkarsh High Performance Computing (HPC) infrastructure**. The computational resources and training support were instrumental in enabling large-scale model training on highly imbalanced solar flare datasets.

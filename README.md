# SolarFlare-Nowcast v2

## Deep Learning-Based Short-Term Solar Flare Forecasting Using GOES X-Ray Observations

Solar flares are among the most energetic phenomena in the Solar System and can significantly impact satellite operations, radio communications, navigation systems, and power-grid infrastructure. Accurate short-term forecasting of major solar flares is therefore a critical challenge in operational space weather prediction.

**SolarFlare-Nowcast v2** is a deep learning framework designed to predict the probability of an **M-class or higher solar flare occurring within the next 60 minutes** using historical GOES X-Ray Sensor (XRS) observations.

The project combines domain-inspired feature engineering with a hybrid **CNN-LSTM architecture** to capture both short-term precursor signatures and long-term temporal evolution patterns in solar X-ray flux measurements.

---

# Project Highlights

* Forecasts **M-class and X-class solar flares**
* Uses **10 years of GOES-15 observations (2010–2020)**
* Hybrid **Convolutional Neural Network + LSTM architecture**
* Handles severe class imbalance using **Focal Loss**
* Evaluated using operational forecasting metrics:

  * True Skill Statistic (TSS)
  * Heidke Skill Score (HSS)
  * Probability of Detection (POD)
  * False Alarm Ratio (FAR)
* Supports:

  * Model training
  * Real-time NOAA GOES data ingestion
  * Command-line inference
  * REST API deployment
  * Interactive Streamlit dashboard

---

# Scientific Objective

Given the previous **6 hours (360 minutes)** of GOES X-ray flux observations:

> Predict whether an M-class or stronger solar flare will occur within the next 60 minutes.

This problem is formulated as a binary classification task under highly imbalanced conditions where flare events are rare relative to quiet-Sun periods.

---

# Dataset

## Data Sources

### GOES X-Ray Flux Data

* NOAA National Centers for Environmental Information (NCEI)
* NOAA Space Weather Prediction Center (SWPC)

### Solar Flare Event Catalog

* LMSAL Heliophysics Event Knowledgebase (HEK)

---

## Training Period

| Parameter        | Value       |
| ---------------- | ----------- |
| Satellite        | GOES-15     |
| Time Span        | 2010–2020   |
| Sampling Rate    | 1 minute    |
| Input Window     | 360 minutes |
| Forecast Horizon | 60 minutes  |

---

# Feature Engineering

Five physically motivated channels are generated from raw GOES XRS measurements.

| Feature     | Description                     |
| ----------- | ------------------------------- |
| xrs_short   | log10(0.05–0.4 nm flux)         |
| xrs_long    | log10(0.1–0.8 nm flux)          |
| xrs_ratio   | Spectral hardness proxy         |
| deriv_short | First-order temporal derivative |
| rolling_max | 30-minute rolling maximum       |

These features were selected to capture early precursor behavior commonly observed before major flare events.

---

# Model Architecture

## CNN-LSTM Hybrid Network

Input Shape:

360 × 5

Pipeline:

Input Sequence

↓

1D Convolution Layers

↓

Batch Normalization

↓

ReLU Activation

↓

LSTM Encoder

↓

Fully Connected Layers

↓

Sigmoid Probability Output

### Why CNN-LSTM?

The original implementation used a dense Multi-Layer Perceptron (MLP), which flattened temporal information and treated all measurements as independent features.

The upgraded CNN-LSTM architecture:

* Preserves temporal ordering
* Learns local flux gradients and precursor patterns
* Captures long-range temporal dependencies
* Provides improved representation learning for sequential solar activity data

---

# Training Strategy

| Component         | Configuration                   |
| ----------------- | ------------------------------- |
| Loss Function     | Focal Loss                      |
| Optimizer         | AdamW                           |
| Scheduler         | ReduceLROnPlateau               |
| Mixed Precision   | Automatic Mixed Precision (AMP) |
| Checkpoint Metric | Validation TSS                  |

---

# Evaluation Metrics

Traditional accuracy is not a meaningful metric for flare forecasting due to severe class imbalance.

A model predicting "No Flare" continuously may exceed 98% accuracy while providing zero operational value.

Instead, evaluation is based on:

| Metric | Purpose                                            |
| ------ | -------------------------------------------------- |
| TSS    | Primary operational forecasting metric             |
| HSS    | Skill relative to random chance                    |
| POD    | Fraction of flares successfully detected           |
| FAR    | Fraction of issued warnings that were false alarms |

The model selection criterion is **maximum validation TSS**.

---

# Repository Structure

```text
solarflare-nowcast/

├── train.py
├── infer.py
├── app.py
├── dashboard.py
├── Config.yaml
├── requirements.txt

├── checkpoints/
│   ├── best_model.pt
│   ├── last_model.pt
│   └── training_history.json

└── src/
    ├── dataset.py
    ├── focal_loss.py
    ├── metrics.py
    ├── model.py
    ├── predictor.py
    ├── goes_fetcher.py
    └── goes_loader.py
```

---

# Installation

```bash
git clone https://github.com/karmakar-rahul/solarflare-nowcast.git

cd solarflare-nowcast

pip install -r requirements.txt
```

---

# Model Training

Full Training:

```bash
python train.py
```

Smoke Test:

```bash
python train.py --smoke
```

Custom Configuration:

```bash
python train.py --config Config.yaml
```

---

# Inference

## Live NOAA Data

```bash
python infer.py --live
```

## Local CSV

```bash
python infer.py --csv path/to/goes_data.csv
```

## Custom Threshold

```bash
python infer.py --live --threshold 0.45
```

---

# Interactive Dashboard

Launch Streamlit dashboard:

```bash
streamlit run dashboard.py
```

Features:

* Live GOES monitoring
* Real-time flare probability
* Historical flux visualization
* Operational warning system

---

# REST API

Start FastAPI server:

```bash
uvicorn app:app --reload --port 8000
```

Endpoints:

```text
GET /live
GET /health
```

API Documentation:

```text
http://localhost:8000/docs
```

---

# Hardware Requirements

| Component | Minimum        | Recommended         |
| --------- | -------------- | ------------------- |
| CPU       | Modern x86 CPU | Intel i5 / Ryzen 5+ |
| RAM       | 8 GB           | 16 GB               |
| GPU       | Optional       | RTX 2050 (4 GB)+    |
| Storage   | 5 GB           | 10 GB               |

---

# Current Limitations

* Trained exclusively on GOES-15 observations
* No active-region magnetic field information
* No spatial localization of flare source regions
* Optimized primarily for M/X-class forecasting
* False alarms are expected due to recall-oriented optimization

---

# Future Work

* Integration of SDO/HMI SHARP magnetic parameters
* Transformer-based sequence forecasting models
* Multi-horizon prediction (30 min, 1 hr, 3 hr, 6 hr)
* Probability calibration
* Ensemble forecasting systems
* Explainable AI methods for flare precursor identification

---

# Research & Technical Contributions

This project demonstrates:

* Time-series forecasting
* Deep learning for scientific data
* Class-imbalance handling
* Operational model evaluation
* Scientific machine learning
* End-to-end ML deployment

---

# Acknowledgements

This work was developed using the **PARAM Utkarsh High Performance Computing (HPC) Infrastructure** provided by **CDAC India (Centre for Development of Advanced Computing)**.

The computational resources and HPC environment made large-scale model development, experimentation, and training possible.

Special thanks to:

* CDAC India
* PARAM Utkarsh HPC Facility
* NOAA NCEI
* NOAA SWPC
* LMSAL HEK

---

# Author

## Rahul Karmakar

**M.Sc. Physics (Astrophysics)**
Assam University, Silchar

Research Interests:

* Space Weather Forecasting
* Solar Physics
* Machine Learning for Scientific Applications
* High Performance Computing
* Computational Astrophysics

---

## License

This repository is intended for research, educational, and portfolio purposes.

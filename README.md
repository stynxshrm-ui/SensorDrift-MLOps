# **SensorDrift-MLOps**

**Real-time wearable sensor health monitoring with drift detection and automated retraining.**

---

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Architecture](#architecture)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Drift Detection](#drift-detection)
8. [Retraining & Model Management](#retraining--model-management)
9. [Dashboard](#dashboard)
10. [Reproducibility](#reproducibility)
11. [Future Work](#future-work)
12. [License](#license)

---

## **Project Overview**

This project simulates a **wearable health monitoring system** that collects vitals (heart rate, SpO₂, steps) and predicts stress/fatigue or detects anomalies. It includes:

* Real-time sensor streaming
* Drift detection in sensor data
* Automatic model retraining
* Monitoring via a dashboard

---

## **Features**

* Real-time prediction API using **FastAPI**
* Sensor distribution and drift tracking (**KL divergence**, **PSI**)
* Automated retraining pipeline with **MLflow/DVC**
* Containerized reproducible environment (**Docker**)
* Dashboard for live metrics and alerts (**Streamlit / Dash**)

---

## **Dataset**

* Options used:

  * [MIMIC-III](https://mimic.mit.edu/) – ICU vitals
  * [WESAD / PhysioNet](https://physionet.org/) – Wearable sensor data for stress detection
* Preprocessing includes resampling, normalization, and feature engineering (HRV metrics, step trends, SpO₂ deviations)

---

## **Architecture**

```
[Wearable Sensors] → [Streaming API] → [ML Model] → [Prediction/Alert]
                                      ↓
                              [Drift Detection] → [Retraining Pipeline]
                                      ↓
                                [Dashboard & Logs]
```

---

## **Installation**

```bash
# Clone repo
git clone https://github.com/<username>/sensor-drift-mlops.git
cd sensor-drift-mlops

# Build Docker container
docker build -t sensor-drift-mlops .

# Run API server
docker run -p 8000:8000 sensor-drift-mlops
```

---

## **Usage**

* **API Endpoint**: `/predict`

  * POST JSON with sensor readings
  * Returns predicted stress/fatigue or anomaly score

* **Simulated streaming**: `python src/stream_simulator.py`

---

## **Drift Detection**

* Tracks feature distribution changes using:

  * KL Divergence
  * Population Stability Index (PSI)
* Alerts when drift exceeds thresholds

---

## **Retraining & Model Management**

* Triggered automatically or manually
* Uses latest sensor batches to update model
* Versions tracked with **MLflow/DVC**
* Old versions retained for rollback

---

## **Dashboard**

* Live plots of:

  * Sensor readings
  * Drift metrics over time
  * Model performance and retraining events

---

## **Reproducibility**

* Fully containerized with **Docker**
* All preprocessing, training, and evaluation scripts included
* Can reproduce results with:

```bash
docker run -it sensor-drift-mlops bash
python notebooks/reproduce_pipeline.ipynb
```

---

## **Future Work**

* Multi-patient streaming and federated learning
* Integration with real wearable devices (Fitbit, Apple Watch)
* Advanced drift detection (adversarial or unsupervised methods)

---

## **License**

MIT License – see `LICENSE` file

---


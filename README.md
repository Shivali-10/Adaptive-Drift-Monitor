# 🛡️ Model Monitoring and Drift Simulation
### Enterprise-grade ML Model Monitoring & Drift Detection Platform

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## 📌 Overview

**Model Monitoring and Drift Simulation** is an end-to-end ML monitoring platform that detects when a machine learning model's performance degrades due to data drift — and automatically retrains the model to restore accuracy.

Built on the **Credit Card Fraud Detection** dataset (284,807 transactions), this platform simulates real-world production scenarios where incoming data changes over time, causing model performance to degrade silently.

> 💡 **Key Insight:** Without drift monitoring, 1 in 8 fraudulent transactions would go undetected by Week 4. Our platform catches this automatically and self-heals.

---

## 🎯 What Problem Does It Solve?

In production ML systems, data distributions change over time — this is called **data drift**. When drift occurs:
- Model accuracy silently degrades
- Predictions become unreliable
- Business decisions are made on faulty outputs

This platform **detects drift early** using 3 statistical tests and **automatically retrains** the model when thresholds are exceeded.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 📤 **Universal Dataset Support** | Works with any uploaded CSV — not just fraud data |
| 🔍 **3 Drift Detection Tests** | KS Test, PSI (Population Stability Index), KL Divergence |
| 📉 **Performance Monitoring** | Tracks AUC, Recall, F1, Precision across time windows |
| 🤖 **Automatic Retraining** | Triggers retraining when drift exceeds threshold |
| 📊 **Before vs After Proof** | Visual AUC comparison before and after retraining |
| 🎨 **Professional Dashboard** | Dark enterprise-themed Streamlit UI with 5 tabs |
| 📄 **Downloadable Report** | Export full analysis as CSV |

---

## 📊 Results

### Model Performance Degradation (Without Retraining)

| Week | Drift Level | AUC | Recall | F1 Score |
|------|------------|-----|--------|----------|
| Week 1 | Baseline | 0.9915 | 0.9831 | 0.9748 |
| Week 2 | Slight | 0.9674 | 0.9348 | 0.9503 |
| Week 3 | Moderate | 0.9883 | 0.9767 | 0.9730 |
| Week 4 | Severe | 0.9255 | 0.8511 | 0.9143 |

> 🚨 **Recall dropped 13.2%** — meaning 1 in 8 frauds were being missed by Week 4!

### After Automatic Retraining

| Week | AUC Before | AUC After | Improvement |
|------|-----------|-----------|-------------|
| Week 2 | 0.9832 | 1.0000 | ▲ +1.7% |
| Week 3 | 0.9955 | 1.0000 | ▲ +0.45% |
| Week 4 | 0.9875 | 1.0000 | ▲ +1.25% |

### Drift Detection Results

| Test | Week 2 | Week 3 | Week 4 |
|------|--------|--------|--------|
| KS Statistic | 0.2776 | 0.5765 | 0.7457 |
| PSI Score | 0.0187 | 0.3518 | 0.8409 |
| KL Divergence | 0.1384 | 0.6989 | 1.2015 |

---

## 🗂️ Project Structure

```
fraud_drift_project/
│
├── 📓 Phase1_EDA.ipynb                     # Exploratory Data Analysis
├── 📓 Phase2_model.ipynb                   # Baseline Model Training
├── 📓 Phase3_Drift_Simulation.ipynb        # Drift Simulation across 4 weeks
├── 📓 Phase4_Drift_Detection.ipynb         # KS Test, PSI, KL Divergence
├── 📓 Phase5_Performance_Monitoring.ipynb  # AUC, Recall, F1 tracking
├── 📓 Phase6_Retraining_Trigger.ipynb      # Auto retraining + proof
│
├── 🐍 dashboard.py                         # Streamlit dashboard (v2.0 Pro)
├── 🤖 fraud_model.pkl                      # Saved trained model
├── 📊 retraining_proof.png                 # Before vs After AUC chart
│
├── 📄 creditcard.csv                       # Original dataset
├── 📄 week1_baseline.csv                   # Week 1 — Baseline
├── 📄 week2_drift.csv                      # Week 2 — Slight drift
├── 📄 week3_drift.csv                      # Week 3 — Moderate drift
└── 📄 week4_drift.csv                      # Week 4 — Severe drift
```

---

## 🔧 Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **ML** | Scikit-learn, Random Forest Classifier |
| **Imbalance Handling** | Imbalanced-learn (SMOTE) |
| **Drift Detection** | SciPy (KS Test), NumPy (PSI, KL Divergence) |
| **Dashboard** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Model Saving** | Joblib |
| **Version Control** | Git, GitHub |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Shivali-10/Adaptive-Drift-Monitor.git
cd Adaptive-Drift-Monitor
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn imbalanced-learn scipy matplotlib seaborn joblib
```

### 3. Run the Streamlit dashboard
```bash
streamlit run dashboard.py
```

### 4. Open in browser
```
http://localhost:8501
```

### 5. Use the dashboard
- Upload any CSV dataset
- Select your target column
- Click **Run Full Analysis**
- Explore all 5 tabs

---

## 📓 Run Notebooks in Order

```
Phase1_EDA.ipynb
      ↓
Phase2_model.ipynb
      ↓
Phase3_Drift_Simulation.ipynb
      ↓
Phase4_Drift_Detection.ipynb
      ↓
Phase5_Performance_Monitoring.ipynb
      ↓
Phase6_Retraining_Trigger.ipynb
```

---

## 🧠 How It Works

```
1. TRAIN          →  Random Forest trained on Week 1 baseline data
2. SIMULATE       →  Gaussian noise injected to simulate drift over 4 weeks
3. DETECT         →  KS Test, PSI and KL Divergence measure distribution shift
4. MONITOR        →  AUC, Recall, F1, Precision tracked per week
5. TRIGGER        →  Alert fired when KS > 0.1 or AUC < 0.95
6. RETRAIN        →  Model automatically retrained on drifted week's data
7. PROVE          →  Before vs After AUC chart shows recovery to 1.0000
```

---

## 📋 Drift Detection Thresholds

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| KS Statistic | > 0.1 | Distribution significantly shifted |
| PSI Score | > 0.2 | Major population shift detected |
| AUC Score | < 0.95 | Model performance degraded |

---

## 👩‍💻 Authors

| Name | GitHub |
|------|--------|
| Vaishali | [@Vaishali-1234](https://github.com/Vaishali-1234) |
| Shivali | [@Shivali-10](https://github.com/Shivali-10) |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset: [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Machine Learning Mastery — drift detection concepts
- Streamlit documentation

---

<p align="center">
  Built with ❤️ by Shivali & Vaishali &nbsp;|&nbsp; Data Science Project 2025
</p>

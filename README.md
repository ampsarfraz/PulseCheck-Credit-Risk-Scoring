# PulseCheck: Instant, Explainable Credit Risk Scoring

A Streamlit application demonstrating instant credit risk scoring with explainable AI using XGBoost and SHAP.

## 🎯 Project Overview

PulseCheck is a dual-view credit risk scoring system that provides:
- **Customer Portal**: Simple loan application with instant decisions
- **Bank Officer Dashboard**: Detailed risk analysis with SHAP explanations

## 🚀 Quick Start

### Prerequisites
```bash
pip install streamlit pandas numpy duckdb joblib shap plotly
```

### Running the Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📊 Features

### Customer Portal
- Simple loan application form
- Instant eligibility decision
- Clear explanations in plain English
- Professional banking interface

### Bank Officer Dashboard
- Login with `admin/admin`
- View all applications with filters
- SHAP-based explainability for each decision
- Portfolio analytics and model performance metrics
- Manual override capability with audit trail

## 🤖 Model Performance

- **Algorithm**: XGBoost (Gradient Boosting)
- **ROC-AUC Score**: 0.9501
- **F1-Score**: 0.83
- **Training Dataset**: 32,581 loan applications

## 📁 Project Structure

```
PulseCheck-Credit-Risk-Scoring/
├── app.py                  # Main Streamlit application
├── xgboost_model.pkl       # Trained XGBoost model
├── scaler.pkl              # Feature scaler
├── data/                   # DuckDB database storage
│   └── applications.duckdb # Application database
├── Project_Models.ipynb    # Model training notebook
├── PulseCheck.ipynb        # Data exploration notebook
└── README.md               # This file
```

## 🎓 Academic Context

This project was developed for MS DSP 422 at Northwestern University, demonstrating:
- Binary classification for loan default prediction
- Handling class imbalance (22% default rate)
- Explainable AI implementation using SHAP
- Professional UI/UX design for financial services

## 💡 Key Technologies

- **Frontend**: Streamlit
- **ML Model**: XGBoost
- **Explainability**: SHAP
- **Database**: DuckDB
- **Visualization**: Plotly

## 📝 License

MIT License - See LICENSE file for details

---

**PulseCheck Credit Risk Scoring System**  
Version 2.0 | Powered by XGBoost & SHAP

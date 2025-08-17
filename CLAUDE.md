# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PulseCheck** - Academic credit risk scoring system for MS DSP 422 at Northwestern University
- Binary classification for loan default prediction (0=no default, 1=default)
- 32,581 loan applications with 12 features  
- 22% default rate (class imbalance)
- Focus on explainable AI using SHAP

## Key Commands

```bash
# Run Jupyter notebooks
jupyter notebook PulseCheck.ipynb
jupyter notebook Project_Models.ipynb

# Install dependencies (create requirements.txt first)
pip install pandas numpy scikit-learn xgboost lightgbm tensorflow shap matplotlib seaborn joblib

# For model training
python -m ipykernel install --user --name pulsecheck
jupyter lab

# Future Streamlit app (not yet implemented)
streamlit run app.py
```

## Data Processing Pipeline

### 1. Missing Value Handling (CRITICAL ORDER)
```python
# Must handle these specific columns
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)  # 895 missing
df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)         # 3116 missing
```

### 2. Feature Engineering Sequence
1. Create ratio features (`income_to_loan_ratio`)
2. Bin employment length (`emp_length_bin`)
3. Add interaction terms (`income_x_credit_length`)
4. Add polynomial features (`loan_percent_squared`)
5. One-hot encode with `drop_first=True`
6. Scale features using `StandardScaler`

### 3. Train/Test Split
```python
# Always use stratification for imbalanced dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

## Model Architecture

### Current Models (in Project_Models.ipynb)
1. **Logistic Regression** - Baseline, interpretable (ROC-AUC: 0.86)
2. **XGBoost** - Primary model (ROC-AUC: 0.94)
3. **LightGBM** - Fast alternative (ROC-AUC: 0.94)
4. **SVM** - Non-tree-based comparison (ROC-AUC: 0.89)
5. **1D CNN** - Deep learning approach (ROC-AUC: 0.91)
6. **ANN** - Simple neural network with class weights

### Model Files Generated
- `logistic_regression_model.pkl`
- `xgboost_model.pkl`
- `lightgbm_model.pkl`
- `svm_model.pkl`
- `cnn_model.keras`
- `ann_model_simple.keras`
- `scaler.pkl`

## Data Schema

### Features (Original 12)
```python
# Personal
person_age              # int64
person_income          # int64
person_home_ownership  # object: RENT, OWN, MORTGAGE, OTHER
person_emp_length      # float64 (has missing values)

# Loan
loan_intent    # object: PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION
loan_grade     # object: A-G
loan_amnt      # int64
loan_int_rate  # float64 (has missing values)
loan_percent_income  # float64

# Credit History
cb_person_default_on_file    # object: Y/N
cb_person_cred_hist_length  # int64

# Target
loan_status  # int64: 0=no default, 1=default
```

### Engineered Features
- `income_to_loan_ratio = person_income / (loan_amnt + 1)`
- `emp_length_bin`: Categorical bins (0-2yrs, 3-7yrs, 7-20yrs, 20yrs+, Unknown)
- `income_x_credit_length`: Interaction term
- `loan_percent_squared`: Polynomial feature

## Critical Implementation Notes

### Class Imbalance Handling
- Use `stratify=y` in train_test_split
- For neural networks: compute class weights
- Primary metrics: ROC-AUC, Precision-Recall AUC
- Avoid accuracy as primary metric

### Model-Specific Requirements
- **Tree models** (XGBoost, LightGBM): No scaling needed
- **Linear/Neural models**: Require StandardScaler
- **SVM**: Requires scaling, use `probability=True` for predict_proba

### Hyperparameter Tuning Patterns
```python
# GridSearchCV template used across models
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring='roc_auc',
    verbose=1,
    n_jobs=-1
)
```

## Common Pitfalls to Avoid

1. **Don't forget stratification** in train/test split
2. **Don't scale before splitting** (data leakage)
3. **Don't drop NaN rows in production** - use imputation instead
4. **Always use drop_first=True** in one-hot encoding
5. **Check for multicollinearity** when adding interaction terms

## Evaluation Metrics Priority

1. **ROC-AUC Score** - Primary metric (target > 0.85)
2. **F1-Score** - For positive class (default prediction)
3. **Precision-Recall AUC** - Important for imbalanced data
4. **Confusion Matrix** - Check false positive rate (regulatory concern)

## SHAP Integration (Not Yet Implemented)

```python
# Pattern for tree-based models
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For neural networks
explainer = shap.DeepExplainer(model, X_train[:100])
```

## Performance Benchmarks

| Model | ROC-AUC | F1-Score | Best Parameters |
|-------|---------|----------|-----------------|
| Logistic Regression | 0.8617 | 0.62 | C=0.1, penalty='l1' |
| XGBoost | 0.9429 | 0.82 | learning_rate=0.1, max_depth=7, n_estimators=200 |
| LightGBM | 0.9434 | 0.82 | learning_rate=0.1, max_depth=10, n_estimators=200 |
| SVM | 0.8893 | 0.74 | C=1, gamma='scale', kernel='rbf' |
| 1D CNN | 0.9113 | 0.74 | epochs=10, batch_size=32 |

## Deployment Considerations (Future)

- Model files are ~1-10MB each (suitable for API deployment)
- Inference time < 100ms required for single prediction
- SHAP explanation generation target < 2 seconds
- Consider model versioning for A/B testing
- Implement input validation for all 12 features

## Development Workflow

1. Load and explore data in `PulseCheck.ipynb`
2. Train models in `Project_Models.ipynb`
3. Save best model as `.pkl` or `.keras`
4. Future: Build Streamlit app for deployment
5. Future: Add SHAP visualizations to app

## File Organization

```
/PulseCheck-Credit-Risk-Scoring/
├── data/
│   └── credit_risk_dataset.csv     # Original dataset
├── PulseCheck.ipynb                # EDA and feature engineering
├── Project_Models.ipynb            # Model training and evaluation
├── *.pkl, *.keras                  # Saved models (generated)
└── app.py                          # Future Streamlit application
```
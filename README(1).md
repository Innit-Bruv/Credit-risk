
# 🧠 Credit Risk Prediction with Explainable ML

## 🔍 Overview

This project predicts the likelihood of loan default using credit application data. It leverages machine learning models, particularly **XGBoost**, and explains predictions using **SHAP values** for transparency and interpretability.

Built from scratch to **learn machine learning by doing**, this project focuses on:
- Feature engineering & cleaning real-world datasets
- Training ML models on imbalanced data
- Explaining black-box models (XAI)
- Evaluating model performance meaningfully

---

## 📁 Project Structure

```
credit-risk-project/
│
├── data/                      # Raw and cleaned datasets
│   └── application_train_cleaned.csv
│
├── notebooks/                 # Jupyter notebooks for each phase
│   ├── phase1_data_cleaning.ipynb
│   ├── phase2_logistic_regression.ipynb
│   └── phase3_xgboost_shap.ipynb
│
├── models/                    # Trained models (optional, currently empty)
│
├── src/                       # (Optional) For modular Python code
│
├── README.md                  # You are here
└── requirements.txt           # Python package dependencies
```

---

## 📊 Technologies Used

- **Python** (NumPy, pandas, matplotlib, seaborn)
- **Scikit-learn** (train/test splits, metrics, logistic regression)
- **XGBoost** (gradient boosting classifier)
- **SHAP** (explainable AI for model interpretation)
- **Jupyter Notebooks** (for iterative development and analysis)

---

## 📈 Results

- **ROC AUC Score**: ~0.73 (with XGBoost)
- Optimal threshold tuned for **recall** of minority class (defaults)
- SHAP explained feature impact — top drivers: `EXT_SOURCE_3`, `AMT_GOODS_PRICE`, `DAYS_EMPLOYED`

---

## 📦 Setup Instructions

```bash
git clone https://github.com/yourusername/credit-risk-project.git
cd credit-risk-project
python -m venv credit_env
source credit_env/bin/activate        # or .\credit_env\Scripts\activate (Windows)
pip install -r requirements.txt
```

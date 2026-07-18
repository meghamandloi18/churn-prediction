# Customer Churn Prediction

Predicts whether a telecom customer is likely to churn (cancel their subscription) based on account, billing, and service usage attributes — the kind of problem subscription businesses (telecom, SaaS, streaming) solve to prioritize retention efforts.

## Problem
Given customer account data (tenure, contract type, monthly charges, support calls, etc.), predict whether a customer will churn, so the business can proactively target at-risk customers with retention offers.

## Dataset
`customer_churn.csv` — 2,000 customer records with realistic behavioral patterns (e.g. month-to-month customers and those with higher charges churn more, matching real-world telecom churn trends). Generated with `generate_data.py` for reproducibility; the same pipeline works unchanged on the well-known [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) if you want to swap in real data.

## Approach
1. **EDA** — churn distribution, churn vs. tenure/contract type/monthly charges, correlation heatmap.
2. **Preprocessing** — encode categorical features, scale numeric features, train/test split (80/20, stratified).
3. **Modeling** — trained and compared Logistic Regression and Random Forest classifiers.
4. **Evaluation** — accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, ROC curve.
5. **Feature importance** — identified the strongest drivers of churn.

## Results
| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.72 | 0.67 | 0.67 | 0.67 | ~0.78 |
| Random Forest | 0.68 | 0.63 | 0.60 | 0.61 | 0.74 |

Logistic Regression outperformed Random Forest on every metric, likely because the underlying churn signal in this dataset is closer to linear — a good reminder that a more complex model isn't always the better choice.

Contract type, monthly charges, and tenure were the strongest predictors of churn — month-to-month customers with high charges and short tenure are the highest-risk segment.

## Tech stack
Python · pandas · NumPy · scikit-learn · matplotlib · seaborn

## How to run
```bash
pip install -r requirements.txt
python generate_data.py              # creates customer_churn.csv
jupyter notebook customer_churn_prediction.ipynb
```

## Files
- `generate_data.py` — creates the dataset
- `customer_churn_prediction.ipynb` — full analysis notebook (EDA → modeling → evaluation)
- `app.py` — interactive Streamlit app (live prediction + EDA + model comparison)
- `customer_churn.csv` — dataset
- `requirements.txt` — dependencies

## Live demo
Deployed on Streamlit Community Cloud: {https://churn-prediction-he2kupyqfpzddkjhl3wksc.streamlit.app

Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Next steps
- Handle class imbalance with SMOTE
- Try gradient-boosted models (XGBoost/LightGBM)
- Hyperparameter tuning with GridSearchCV

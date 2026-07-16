"""
Generates a synthetic but realistic telecom customer churn dataset.
Signal is deliberately built in (tenure, contract type, monthly charges
drive churn probability) so that EDA and models produce sensible results.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 2000

tenure = rng.integers(0, 72, n)  # months
contract_type = rng.choice(['Month-to-Month', 'One Year', 'Two Year'], n, p=[0.55, 0.25, 0.20])
internet_service = rng.choice(['DSL', 'Fiber Optic', 'No'], n, p=[0.35, 0.45, 0.20])
payment_method = rng.choice(
    ['Electronic Check', 'Mailed Check', 'Bank Transfer', 'Credit Card'], n
)
monthly_charges = np.round(rng.normal(65, 25, n).clip(18, 120), 2)
total_charges = np.round(monthly_charges * tenure * rng.uniform(0.9, 1.0, n), 2)
num_support_calls = rng.poisson(1.5, n)
senior_citizen = rng.choice([0, 1], n, p=[0.84, 0.16])
partner = rng.choice(['Yes', 'No'], n)
paperless_billing = rng.choice(['Yes', 'No'], n, p=[0.6, 0.4])

# Build churn probability from a logistic function of real signal
contract_weight = np.select(
    [contract_type == 'Month-to-Month', contract_type == 'One Year', contract_type == 'Two Year'],
    [1.1, -0.4, -1.3]
)
logit = (
    -1.2
    + contract_weight
    + 0.35 * (monthly_charges - 65) / 25
    - 0.9 * (tenure - 36) / 36
    + 0.25 * num_support_calls
    + 0.3 * (internet_service == 'Fiber Optic')
    + 0.2 * (payment_method == 'Electronic Check')
)
prob_churn = 1 / (1 + np.exp(-logit))
churn = (rng.uniform(0, 1, n) < prob_churn).astype(int)
churn_label = np.where(churn == 1, 'Yes', 'No')

df = pd.DataFrame({
    'CustomerID': [f'CUST{10000+i}' for i in range(n)],
    'Tenure': tenure,
    'ContractType': contract_type,
    'InternetService': internet_service,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'NumSupportCalls': num_support_calls,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'PaperlessBilling': paperless_billing,
    'Churn': churn_label,
})

df.to_csv('customer_churn.csv', index=False)
print(df['Churn'].value_counts(normalize=True))
print(f"Saved {len(df)} rows to customer_churn.csv")

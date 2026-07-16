"""
Customer Churn Prediction — Streamlit App
Trains a model on startup (cached) and lets a user enter a customer's
details to get a live churn prediction, plus shows the underlying EDA.
"""
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("customer_churn.csv")


@st.cache_resource
def train_models(df):
    data = df.copy()
    for col in ["Partner", "PaperlessBilling"]:
        data[col] = data[col].map({"Yes": 1, "No": 0})

    cat_cols = ["ContractType", "InternetService", "PaymentMethod"]
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    le = LabelEncoder()
    data["Churn"] = le.fit_transform(data["Churn"])

    feature_cols = [c for c in data.columns if c not in ["CustomerID", "Churn"]]
    X = data[feature_cols]
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    num_cols = ["Tenure", "MonthlyCharges", "TotalCharges"]
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "auc": roc_auc_score(y_test, probs),
            "confusion_matrix": confusion_matrix(y_test, preds),
        }

    return results, scaler, feature_cols, num_cols


df = load_data()
results, scaler, feature_cols, num_cols = train_models(df)

st.title("📉 Customer Churn Prediction")
st.caption(
    "Predicts whether a telecom customer is likely to churn, based on account and usage attributes."
)

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Data Explorer", "🧪 Model Performance"])

# ---------------- TAB 1: Live prediction ----------------
with tab1:
    st.subheader("Enter customer details")
    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges))
        num_support_calls = st.slider("Support calls (last year)", 0, 10, 1)

    with col2:
        contract_type = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])
        payment_method = st.selectbox(
            "Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"]
        )

    with col3:
        senior_citizen = st.radio("Senior Citizen", ["No", "Yes"], horizontal=True)
        partner = st.radio("Has Partner", ["No", "Yes"], horizontal=True)
        paperless_billing = st.radio("Paperless Billing", ["No", "Yes"], horizontal=True)

    model_choice = st.selectbox("Model", list(results.keys()))

    if st.button("Predict churn", type="primary"):
        row = {c: 0 for c in feature_cols}
        row["Tenure"] = tenure
        row["MonthlyCharges"] = monthly_charges
        row["TotalCharges"] = total_charges
        row["NumSupportCalls"] = num_support_calls
        row["SeniorCitizen"] = 1 if senior_citizen == "Yes" else 0
        row["Partner"] = 1 if partner == "Yes" else 0
        row["PaperlessBilling"] = 1 if paperless_billing == "Yes" else 0

        if f"ContractType_{contract_type}" in row:
            row[f"ContractType_{contract_type}"] = 1
        if f"InternetService_{internet_service}" in row:
            row[f"InternetService_{internet_service}"] = 1
        if f"PaymentMethod_{payment_method}" in row:
            row[f"PaymentMethod_{payment_method}"] = 1

        input_df = pd.DataFrame([row])[feature_cols]
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        model = results[model_choice]["model"]
        prob = model.predict_proba(input_df)[0, 1]
        pred = "Likely to CHURN" if prob >= 0.5 else "Likely to STAY"

        st.metric("Prediction", pred, f"{prob:.0%} churn probability")
        st.progress(min(max(prob, 0.0), 1.0))

# ---------------- TAB 2: EDA ----------------
with tab2:
    st.subheader("Dataset overview")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"{len(df)} customers, churn rate: {(df['Churn'] == 'Yes').mean():.1%}")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax)
        ax.set_title("Monthly Charges vs Churn")
        st.pyplot(fig)
    with c2:
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df, x="Churn", y="Tenure", ax=ax2)
        ax2.set_title("Tenure vs Churn")
        st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="ContractType", hue="Churn", ax=ax3)
    ax3.set_title("Contract Type vs Churn")
    st.pyplot(fig3)

# ---------------- TAB 3: Model performance ----------------
with tab3:
    st.subheader("Model comparison")
    perf_df = pd.DataFrame(
        {
            name: {
                "Accuracy": f"{r['accuracy']:.2%}",
                "F1-score": f"{r['f1']:.2f}",
                "ROC-AUC": f"{r['auc']:.2f}",
            }
            for name, r in results.items()
        }
    ).T
    st.table(perf_df)

    chosen = st.selectbox("View confusion matrix for", list(results.keys()), key="cm_select")
    fig4, ax4 = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        results[chosen]["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax4,
    )
    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("Actual")
    st.pyplot(fig4)

import streamlit as st
import pandas as pd
import mlflow
from pathlib import Path

# ============================
# Config
# ============================


MODEL_DIR = Path(__file__).parent.parent.parent / "src" / "serving" / "model"
THRESHOLD = 0.5  # replace with your tuned threshold from find_best_threshold

EMP_LENGTH_MAP = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10
}


@st.cache_resource
def load_model():
    """Loads the MLflow pyfunc model once and caches it across reruns/sessions."""
    return mlflow.pyfunc.load_model(str(MODEL_DIR))


def predict(input_df: pd.DataFrame) -> dict:
    model = load_model()
    proba = model.predict(input_df)  # returns probability, since logged with pyfunc_predict_fn="predict_proba"
    proba_value = float(proba[0]) if hasattr(proba, "__len__") else float(proba)
    prediction = int(proba_value >= THRESHOLD)
    return {"prediction": prediction, "probability": proba_value}


# ============================
# UI
# ============================

st.set_page_config(page_title="Lloyd Bank Risk Portal", layout="centered")

st.title("🏦 Lloyd Bank Loan Assessment")
st.info("This app runs the trained model directly — no separate backend required.")


def reset_form():
    st.session_state["form_data"] = {}
    st.rerun()


with st.form("loan_application", clear_on_submit=False):
    tab1, tab2, tab3 = st.tabs(["💵 Loan & Identity", "💼 Employment", "📊 Credit History"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            loan_amnt = st.number_input("Loan Amount ($)", value=15000.0)
            term = st.selectbox("Term", [" 36 months", " 60 months"])
            int_rate = st.number_input("Interest Rate (as decimal, e.g. 0.12 for 12%)", value=0.12)
            installment = st.number_input("Monthly Installment ($)", value=450.0)
        with col2:
            purpose = st.selectbox("Loan Purpose", ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "other"])

    with tab2:
        emp_length_label = st.selectbox(
            "Employment Length",
            ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years',
             '6 years', '7 years', '8 years', '9 years', '10+ years']
        )
        annual_inc = st.number_input("Annual Income ($)", value=65000.0)
        home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "ANY"])

    with tab3:
        c1, c2, c3 = st.columns(3)
        with c1:
            avg_cur_bal = st.number_input("Avg Current Balance", value=15000.0)
            max_bal_bc = st.number_input("Max Balance (BC)", value=5000.0)
            total_bal_ex_mort = st.number_input("Total Bal (Ex Mort)", value=30000.0)
            mort_acc = st.number_input("Mortgage Accounts", value=1)
            open_acc = st.number_input("Open Accounts", value=12)
        with c2:
            inq_last_12m = st.number_input("Inquiries (12m)", value=0)
            no_delinq_history = st.checkbox("No delinquency history", value=True)
            mths_since_last_delinq = None if no_delinq_history else st.number_input(
                "Months Since Delinq", value=12, min_value=0
            )
            pub_rec_bankruptcies = st.number_input("Bankruptcies", value=0)
            percent_bc_gt_75 = st.slider("% Cards Over 75% Limit", 0, 100, 25)
            total_acc = st.number_input("Total Accounts", value=20)
        with c3:
            mo_sin_old_il_acct = st.number_input("Oldest IL Acct (Mo)", value=120)
            mo_sin_old_rev_tl_op = st.number_input("Oldest Rev Acct (Mo)", value=180)
            mo_sin_rcnt_rev_tl_op = st.number_input("Recent Rev Acct (Mo)", value=12)
            mo_sin_rcnt_tl = st.number_input("Recent Account (Mo)", value=6)
            num_bc_tl = st.number_input("Num BC Accounts", value=5)
            num_il_tl = st.number_input("Num IL Accounts", value=8)
            num_op_rev_tl = st.number_input("Num Open Rev Accounts", value=10)
            num_tl_90g_dpd_24m = st.number_input("90 Days Late (24m)", value=0)
            num_tl_op_past_12m = st.number_input("Acc Opened (12m)", value=2)

    submit = st.form_submit_button("Run Risk Assessment")


if st.button("🔄 Clear Form & Reset"):
    reset_form()


if submit:
    input_data = {
        "annual_inc": annual_inc,
        "emp_length": EMP_LENGTH_MAP[emp_length_label],
        "home_ownership": home_ownership,
        "installment": installment,
        "loan_amnt": loan_amnt,
        "purpose": purpose,
        "term": term,
        "int_rate": int_rate,
        "avg_cur_bal": avg_cur_bal,
        "inq_last_12m": inq_last_12m,
        "max_bal_bc": max_bal_bc,
        "mo_sin_old_il_acct": mo_sin_old_il_acct,
        "mo_sin_old_rev_tl_op": mo_sin_old_rev_tl_op,
        "mo_sin_rcnt_rev_tl_op": mo_sin_rcnt_rev_tl_op,
        "mo_sin_rcnt_tl": mo_sin_rcnt_tl,
        "mort_acc": mort_acc,
        "mths_since_last_delinq": mths_since_last_delinq,
        "num_bc_tl": num_bc_tl,
        "num_il_tl": num_il_tl,
        "num_op_rev_tl": num_op_rev_tl,
        "num_tl_90g_dpd_24m": num_tl_90g_dpd_24m,
        "num_tl_op_past_12m": num_tl_op_past_12m,
        "open_acc": open_acc,
        "percent_bc_gt_75": percent_bc_gt_75,
        "pub_rec_bankruptcies": pub_rec_bankruptcies,
        "total_acc": total_acc,
        "total_bal_ex_mort": total_bal_ex_mort,
    }

    try:
        with st.spinner("Model is analyzing risk..."):
            input_df = pd.DataFrame([input_data])
            result = predict(input_df)

        if result["prediction"] == 1:
            st.error(f"🔴 Likely to default — probability: {result['probability']:.1%}")
        else:
            st.success(f"🟢 Not likely to default — probability: {result['probability']:.1%}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
import requests

# If running uvicorn locally: http://127.0.0.1:8000
# If running in Docker: http://localhost:8000
url = "http://127.0.0.1:8000/predict"


sample_data = {
    "addr_state": "NY",
    "emp_length": "5 years",
    "emp_title": "Manager",
    "home_ownership": "MORTGAGE",
    "purpose": "debt_consolidation",
    "term": " 36 months",
    "annual_inc": 65000.0,
    "installment": 450.0,
    "loan_amnt": 15000.0,
    "int_rate": 12.0,
    "avg_cur_bal": 15000.0,
    "inq_last_12m": 0.0,
    "max_bal_bc": 5000.0,
    "mo_sin_old_il_acct": 120.0,
    "mo_sin_old_rev_tl_op": 180.0,
    "mo_sin_rcnt_rev_tl_op": 12.0,
    "mo_sin_rcnt_tl": 6.0,
    "mort_acc": 1.0,
    "mths_since_last_delinq": -1.0,
    "num_bc_tl": 5.0,
    "num_il_tl": 8.0,
    "num_op_rev_tl": 10.0,
    "num_tl_90g_dpd_24m": 0.0,
    "num_tl_op_past_12m": 2.0,
    "open_acc": 12.0,
    "percent_bc_gt_75": 25.0,
    "pub_rec_bankruptcies": 0.0,
    "total_acc": 20.0,
    "total_bal_ex_mort": 30000.0
}

try:
    response = requests.post(url, json=sample_data)
    print(f"Status Code: {response.status_code}")
    print(f"Prediction: {response.json()}")
except Exception as e:
    print(f"Connection failed: {e}")
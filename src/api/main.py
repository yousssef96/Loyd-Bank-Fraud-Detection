from fastapi import FastAPI
from pydantic import BaseModel
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.serving.inference import predict  # Core ML inference logic

# Initialize FastAPI application
app = FastAPI(
    title="Lloyd Bank Loan Prediction API",
    description="ML API for predicting customer loan default in the Banking industry",
    version="1.0.0"
)

# === HEALTH CHECK ENDPOINT ===
# CRITICAL: Required for AWS Application Load Balancer health checks
@app.get("/")
def root():
    """
    Health check endpoint for monitoring and load balancer health checks.
    """
    return {"status": "ok"}

# === REQUEST DATA SCHEMA ===
# Pydantic model for automatic validation and API documentation
# === REQUEST DATA SCHEMA ===
class CustomerLoanData(BaseModel):
    """
    Schema representing a Lloyd Bank loan application.
    Includes all categorical and numeric features identified in your dataset.
    """
    
    # Categorical Features
    addr_state: str
    emp_length: str
    emp_title: str
    home_ownership: str
    purpose: str
    term: str
    
    # Numeric Features (Critical for math operations)
    annual_inc: float
    installment: float
    loan_amnt: float
    int_rate: float
    avg_cur_bal: float
    inq_last_12m: float
    max_bal_bc: float
    mo_sin_old_il_acct: float
    mo_sin_old_rev_tl_op: float
    mo_sin_rcnt_rev_tl_op: float
    mo_sin_rcnt_tl: float
    mort_acc: float
    mths_since_last_delinq: float
    num_bc_tl: float
    num_il_tl: float
    num_op_rev_tl: float
    num_tl_90g_dpd_24m: float
    num_tl_op_past_12m: float
    open_acc: float
    percent_bc_gt_75: float
    pub_rec_bankruptcies: float
    total_acc: float
    total_bal_ex_mort: float

# === MAIN PREDICTION API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: CustomerLoanData):
    """
    Endpoint for real-time loan default risk assessment.
    
    1. Validates incoming loan application data.
    2. Passes data to the _serve_transform and model pipeline.
    3. Returns specific Lloyd Bank risk status.
    
    Expected Response:
    - {"prediction": "Likely to default"} or {"prediction": "Not likely to default"}
    - {"error": "error_message"} if prediction fails
    """
    try:
        # Convert Pydantic model to dict and call inference pipeline
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        # Return error details for debugging (consider logging in production)
        return {"error": str(e)}


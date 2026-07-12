from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from src.serving.inference import predict, load_model
from src.utils.logger import logger

app = FastAPI(title="Loyd Bank Fraud Detection API")


class LoanApplication(BaseModel):
    annual_inc: float
    emp_length: float
    home_ownership: str
    installment: float
    loan_amnt: float
    purpose: str
    term: str
    int_rate: float
    avg_cur_bal: float
    inq_last_12m: float
    max_bal_bc: float
    mo_sin_old_il_acct: float
    mo_sin_old_rev_tl_op: float
    mo_sin_rcnt_rev_tl_op: float
    mo_sin_rcnt_tl: float
    mort_acc: float
    mths_since_last_delinq: Optional[float] = None
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


@app.on_event("startup")
def startup_event():
    load_model()
    logger.info("API startup complete, model ready")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(application: LoanApplication):
    try:
        input_df = pd.DataFrame([application.model_dump()])
        result = predict(input_df)
        return {
            "prediction": result["prediction"][0],
            "probability": result["probability"][0]
        }
    except Exception as e:
        logger.exception("Prediction request failed")
        raise HTTPException(status_code=500, detail=str(e))
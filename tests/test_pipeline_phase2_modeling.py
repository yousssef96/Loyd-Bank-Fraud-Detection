import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from category_encoders import TargetEncoder
import optuna


print("=== Phase 2: Modeling with LightGBM ===")


df = pd.read_excel("data/processed/LGB_loan_processed.xlsx")

# target must be numeric 0/1
if df["loan_status"].dtype == "object":
    df["loan_status"] = df["loan_status"].str.strip().map({'Fully Paid' : 0, 'Charged Off': 1})

assert df["loan_status"].isna().sum() == 0, "loan_status has NaNs"
assert set(df["loan_status"].unique()) <= {0, 1}, "loan_status not 0/1"


X = df.drop(columns=["loan_status"])
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)



def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000), 
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "random_state": 42,
        "importance_type": "gain"
        }
    model = LGBMClassifier(**params, verbosity=-1)
    pipe_lgbm = Pipeline([
        ('encoder', TargetEncoder(cols=['emp_title', 'addr_state'])),
        ('classifier', model)
        ])
    
    cv_scores = cross_val_score(pipe_lgbm,
                                    X_train,
                                    y_train,
                                    cv = 5,
                                    scoring="recall",
                                    n_jobs=-1, 
                                    verbose=0)
        
    avg_recall = cv_scores.mean()
    return avg_recall 

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print("Best Params:", study.best_params)
print("Best Recall:", study.best_value)
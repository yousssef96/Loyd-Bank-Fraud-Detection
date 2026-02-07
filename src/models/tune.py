import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder

def tune_model(X, y):
    """
    Tunes a LlightGBM model using Optuna.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
    """
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
        "random_state": 42,
        "importance_type": "gain"
        }

        model = LGBMClassifier(**params)
        pipe_lgbm = Pipeline([
            ('encoder', TargetEncoder(cols=['emp_title', 'addr_state'])),
            ('classifier', model)
        ])
        scores = cross_val_score(estimator=pipe_lgbm, X=X, y=y, cv=5, scoring="recall")
        return scores.mean()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    
    print("Best Params:", study.best_params)
    return study.best_params
"""
Logistic Regression and XGBoost pipelines for binary and regression.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def get_lr_pipeline(target_type: str) -> Pipeline:
    """LR pipeline: Imputer + StandardScaler + LogisticRegression. Binary only."""
    if target_type != "binary":
        raise ValueError("LogisticRegression only supports target_type='binary'")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def get_xgb_model(target_type: str, **kwargs):
    """XGBoost for binary (XGBClassifier) or regression (XGBRegressor)."""
    import xgboost as xgb
    default_kw = dict(max_depth=6, n_estimators=100, random_state=42)
    default_kw.update(kwargs)
    if target_type == "binary":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            **default_kw,
        )
    if target_type == "regression":
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            **default_kw,
        )
    raise ValueError(f"target_type must be 'binary' or 'regression', got {target_type}")


def get_lgbm_model(target_type: str, **kwargs):
    """LightGBM for binary or regression (Tweedie for count)."""
    import lightgbm as lgb
    default_kw = dict(max_depth=6, n_estimators=100, random_state=42, verbosity=-1, n_jobs=1)
    default_kw.update(kwargs)
    if target_type == "binary":
        return lgb.LGBMClassifier(objective="binary", **default_kw)
    if target_type == "regression":
        return lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5, **default_kw)
    raise ValueError(f"target_type must be 'binary' or 'regression', got {target_type}")

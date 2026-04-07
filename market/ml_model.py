# market/ml_model.py

import xgboost as xgb
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

def get_model_path(coin):
    return f"models/xgb_{coin}.pkl"


def load_model(coin):
    path = get_model_path(coin)
    if os.path.exists(path):
        return joblib.load(path)
    return None


def create_models():
    xgb_model = xgb.XGBRegressor(n_estimators=200)
    rf_model = RandomForestRegressor(n_estimators=100)
    return xgb_model, rf_model


def train_models(X, y):
    xgb_model, rf_model = create_models()

    xgb_model.fit(X, y)
    rf_model.fit(X, y)

    return xgb_model, rf_model
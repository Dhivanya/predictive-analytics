"""
forecaster.py — regression and time-series forecasting models
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _time_index(n: int) -> np.ndarray:
    return np.arange(n).reshape(-1, 1)


# ── Model fitting ──────────────────────────────────────────────────────────────

def fit_linear(y: np.ndarray):
    X = _time_index(len(y))
    model = LinearRegression().fit(X, y)
    return model, "Linear Regression"


def fit_polynomial(y: np.ndarray, degree: int = 2):
    X = _time_index(len(y))
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression()).fit(X, y)
    return model, f"Polynomial Regression (deg={degree})"


def fit_moving_average(y: np.ndarray, window: int = 6):
    """Returns a dict instead of a sklearn model (MA has no sklearn fit)."""
    series = pd.Series(y)
    ma = series.rolling(window=window).mean().values
    # Estimate trend from last two valid MA values
    valid = [v for v in ma if not np.isnan(v)]
    trend = valid[-1] - valid[-2] if len(valid) >= 2 else 0
    return {"ma": ma, "last": valid[-1], "trend": trend, "window": window}, f"Moving Average (window={window})"


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(model_obj, n_history: int, n_forecast: int, model_name: str):
    """Returns fitted values for history + forecast values."""
    X_hist = _time_index(n_history)
    X_fore = np.arange(n_history, n_history + n_forecast).reshape(-1, 1)

    if isinstance(model_obj, dict):                         # Moving average
        fitted = model_obj["ma"]
        last, trend = model_obj["last"], model_obj["trend"]
        forecast = np.array([last + trend * (i + 1) for i in range(n_forecast)])
    else:                                                   # sklearn models
        fitted = model_obj.predict(X_hist)
        forecast = model_obj.predict(X_fore)

    return fitted, forecast


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, MAPE, and R²."""
    mask = ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]

    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mape = np.mean(np.abs((yt - yp) / yt)) * 100
    r2   = r2_score(yt, yp)

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE (%)": round(mape, 4), "R²": round(r2, 4)}


def run_all_models(y: np.ndarray, n_forecast: int = 12) -> dict:
    """Fit all three models and return their metrics for comparison."""
    results = {}

    for name, fit_fn in [("Linear", fit_linear), ("Polynomial", fit_polynomial)]:
        model, label = fit_fn(y)
        fitted, forecast = predict(model, len(y), n_forecast, label)
        metrics = evaluate(y, fitted)
        results[label] = {"fitted": fitted, "forecast": forecast, "metrics": metrics}

    model, label = fit_moving_average(y)
    fitted, forecast = predict(model, len(y), n_forecast, label)
    metrics = evaluate(y, fitted)
    results[label] = {"fitted": fitted, "forecast": forecast, "metrics": metrics}

    return results

"""
data_loader.py — generates and loads synthetic historical datasets
"""
import pandas as pd
import numpy as np


def generate_sales_data(n_months: int = 48, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range(start="2020-01-01", periods=n_months, freq="MS")
    trend = np.linspace(40, 160, n_months)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(n_months) / 12)
    noise = np.random.normal(0, 4, n_months)
    revenue = trend + seasonality + noise
    return pd.DataFrame({"date": dates, "value": revenue.round(2), "dataset": "Monthly Sales Revenue (₹ Lakhs)"})


def generate_temperature_data(n_months: int = 48, seed: int = 7) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range(start="2020-01-01", periods=n_months, freq="MS")
    trend = np.linspace(14, 60, n_months)
    noise = np.random.normal(0, 2.5, n_months)
    anomaly = trend + noise
    return pd.DataFrame({"date": dates, "value": anomaly.round(2), "dataset": "Temperature Anomaly (°C × 100)"})


def generate_user_data(n_months: int = 48, seed: int = 99) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range(start="2020-01-01", periods=n_months, freq="MS")
    base = 10
    growth = np.array([base * (1.06 ** i) for i in range(n_months)])
    noise = np.random.normal(0, growth * 0.03)
    users = growth + noise
    return pd.DataFrame({"date": dates, "value": users.round(2), "dataset": "Monthly Active Users (K)"})


DATASETS = {
    "Monthly Sales Revenue (₹ Lakhs)": generate_sales_data,
    "Temperature Anomaly (°C × 100)": generate_temperature_data,
    "Monthly Active Users (K)": generate_user_data,
}


def load_dataset(name: str) -> pd.DataFrame:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASETS.keys())}")
    return DATASETS[name]()

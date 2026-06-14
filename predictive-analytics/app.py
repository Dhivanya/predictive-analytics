"""
app.py — Predictive Analytics Dashboard (Streamlit)
Run:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import load_dataset, DATASETS
from models.forecaster import (
    fit_linear, fit_polynomial, fit_moving_average,
    predict, evaluate, run_all_models
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Analytics Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Predictive Analytics — Historical Trend Forecasting")
st.markdown("Explore regression and time-series models on synthetic datasets. Adjust parameters in the sidebar.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    dataset_name = st.selectbox("Dataset", list(DATASETS.keys()))
    model_choice = st.selectbox(
        "Forecasting Model",
        ["Linear Regression", "Polynomial Regression (deg=2)", "Moving Average (window=6)", "Compare All Models"]
    )
    n_forecast = st.slider("Forecast Periods (months)", 3, 24, 12)
    show_residuals = st.checkbox("Show residuals plot", value=True)
    show_table = st.checkbox("Show data table", value=False)

    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        "Built with Python · Streamlit · Plotly · scikit-learn\n\n"
        "[GitHub →](https://github.com/Dhivanya)"
    )

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_dataset(dataset_name)
y  = df["value"].values
dates = df["date"].values
n = len(y)
future_dates = pd.date_range(start=df["date"].iloc[-1] + pd.DateOffset(months=1), periods=n_forecast, freq="MS")

unit = dataset_name.split("(")[-1].replace(")", "").strip() if "(" in dataset_name else ""

# ── Fit model(s) ──────────────────────────────────────────────────────────────
COLORS = {"Linear Regression": "#378ADD", "Polynomial Regression (deg=2)": "#639922",
          "Moving Average (window=6)": "#D85A30"}

if model_choice == "Compare All Models":
    all_results = run_all_models(y, n_forecast)
else:
    key_map = {
        "Linear Regression":           (fit_linear,          {}),
        "Polynomial Regression (deg=2)":(fit_polynomial,     {"degree": 2}),
        "Moving Average (window=6)":   (fit_moving_average,  {"window": 6}),
    }
    fit_fn, kwargs = key_map[model_choice]
    model_obj, label = fit_fn(y, **kwargs)
    fitted, forecast = predict(model_obj, n, n_forecast, label)
    metrics = evaluate(y, fitted)

# ── Metric cards ──────────────────────────────────────────────────────────────
if model_choice != "Compare All Models":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score",      f"{metrics['R²']:.4f}",     help="1.0 = perfect fit")
    c2.metric("MAPE",          f"{metrics['MAPE (%)']:.2f}%", help="Mean Absolute Percentage Error")
    c3.metric("MAE",           f"{metrics['MAE']:.2f} {unit}")
    c4.metric("RMSE",          f"{metrics['RMSE']:.2f} {unit}")
    st.markdown("---")

# ── Forecast chart ────────────────────────────────────────────────────────────
st.subheader("🔮 Forecast Chart")

fig = go.Figure()

# Historical trace
fig.add_trace(go.Scatter(
    x=dates, y=y, mode="lines+markers", name="Historical",
    line=dict(color="#378ADD", width=2), marker=dict(size=4)
))

if model_choice == "Compare All Models":
    for label, res in all_results.items():
        color = COLORS.get(label, "#888")
        mask = ~np.isnan(res["fitted"])
        fig.add_trace(go.Scatter(
            x=dates[mask], y=res["fitted"][mask], mode="lines", name=f"{label} (fit)",
            line=dict(color=color, width=1.5, dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=res["forecast"], mode="lines+markers", name=f"{label} (forecast)",
            line=dict(color=color, width=2, dash="dash"), marker=dict(size=5, symbol="diamond")
        ))
else:
    mask = ~np.isnan(fitted)
    fig.add_trace(go.Scatter(
        x=dates[mask], y=fitted[mask], mode="lines", name="Fitted",
        line=dict(color="#639922", width=1.5, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=forecast, mode="lines+markers", name="Forecast",
        line=dict(color="#D85A30", width=2, dash="dash"), marker=dict(size=6, symbol="diamond")
    ))

# Divider between history and forecast
fig.add_vline(x=str(df["date"].iloc[-1]), line_dash="longdash", line_color="gray", opacity=0.4)
fig.add_annotation(x=str(df["date"].iloc[-1]), y=1, yref="paper",
                   text="Forecast →", showarrow=False, xanchor="left",
                   font=dict(size=11, color="gray"))

fig.update_layout(
    height=420, hovermode="x unified",
    xaxis_title="Date", yaxis_title=dataset_name,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
)
st.plotly_chart(fig, use_container_width=True)

# ── Residuals ─────────────────────────────────────────────────────────────────
if show_residuals and model_choice != "Compare All Models":
    st.subheader("📉 Residuals Analysis")

    mask = ~np.isnan(fitted)
    residuals = y[mask] - fitted[mask]

    col1, col2 = st.columns(2)

    with col1:
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=fitted[mask], y=residuals, mode="markers",
            marker=dict(color="#7F77DD", size=7, opacity=0.7),
            name="Residuals"
        ))
        fig_r.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_r.update_layout(
            title="Predicted vs Residual", height=300,
            xaxis_title="Predicted", yaxis_title="Residual",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col2:
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(x=residuals, nbinsx=15, marker_color="#1D9E75", opacity=0.75))
        fig_h.update_layout(
            title="Error Distribution", height=300,
            xaxis_title="Residual", yaxis_title="Count",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_h, use_container_width=True)

# ── Model comparison table ────────────────────────────────────────────────────
if model_choice == "Compare All Models":
    st.subheader("📊 Model Comparison")
    rows = []
    for label, res in all_results.items():
        rows.append({"Model": label, **res["metrics"]})
    comp_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(
        comp_df.style
            .highlight_max(subset=["R²"], color="#c6efce")
            .highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], color="#c6efce"),
        use_container_width=True
    )

# ── Raw data table ────────────────────────────────────────────────────────────
if show_table:
    st.subheader("📋 Raw Data")
    if model_choice != "Compare All Models":
        display_df = df.copy()
        display_df["fitted"] = np.round(fitted, 2)
        display_df["residual"] = np.round(y - fitted, 2)
        st.dataframe(display_df, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Predictive Analytics Project · Built with Streamlit, Plotly, scikit-learn · Dhivanya · github.com/Dhivanya")

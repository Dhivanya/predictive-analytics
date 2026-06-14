# 📈 Predictive Analytics Using Historical Data

A Streamlit web application that forecasts future trends using regression and time-series models on historical datasets.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-blue)

---

## 🚀 Live Demo

https://predictive-analytics-rdag6vo3wsrt9ksk5vvrhm.streamlit.app/

---

## 📌 Features

- **3 Forecasting Models** — Linear Regression, Polynomial Regression (deg=2), Moving Average
- **3 Datasets** — Monthly Sales Revenue, Temperature Anomaly, User Growth
- **4 Accuracy Metrics** — MAE, RMSE, MAPE, R²
- **Interactive Charts** — Plotly-powered forecast + residuals + error distribution
- **Model Comparison** — Side-by-side metrics table with best-model highlighting
- **CLI Script** — `train_and_evaluate.py` for offline batch evaluation + PNG exports

---

## 🗂️ Project Structure

```
predictive-analytics/
├── app.py                    # Streamlit dashboard (main entry point)
├── train_and_evaluate.py     # CLI: train all models, export metrics + plots
├── requirements.txt
├── utils/
│   └── data_loader.py        # Synthetic dataset generators
├── models/
│   └── forecaster.py         # Model fitting, prediction, evaluation
└── output/                   # Auto-created: plots + metrics CSV
```

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/Dhivanya/predictive-analytics.git
cd predictive-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit app
streamlit run app.py

# 4. (Optional) Run CLI batch evaluation
python train_and_evaluate.py
```

---

## 🧠 Models Explained

| Model | How it works | Best for |
|---|---|---|
| Linear Regression | Fits a straight trend line (least squares) | Steady, consistent growth |
| Polynomial Regression | Fits a curve of degree 2 | Accelerating / decelerating trends |
| Moving Average | Smooths noise, extrapolates recent momentum | Noisy or seasonal data |

---

## 📊 Evaluation Metrics

| Metric | Formula | Ideal |
|---|---|---|
| MAE | Mean of |actual − predicted| | Lower is better |
| RMSE | √(Mean of (actual − predicted)²) | Lower is better |
| MAPE | Mean of |(actual − predicted) / actual| × 100 | < 10% is good |
| R² | 1 − SS_res / SS_tot | Closer to 1 |

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit** — interactive web UI
- **Plotly** — interactive charts
- **scikit-learn** — regression models
- **Pandas / NumPy** — data preprocessing
- **Matplotlib** — CLI plot exports

---

## 👤 Author

**Dhivanya** — B.E. ECE, SNS College of Engineering  
[GitHub](https://github.com/Dhivanya)

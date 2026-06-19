# 📈 Predictive Analytics Using Historical Data

A Streamlit web application that forecasts future trends using regression and time-series models on historical datasets.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-blue)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

---

## 🚀 Live Demo

👉(https://YOUR-STREAMLIT-URL.streamlit.app)**

---

## 📌 What This Project Does

This project takes **historical data** (sales, users, temperature) and:
- Trains **Machine Learning models** on past data
- **Forecasts / predicts** future values for next 6–18 months
- Shows **accuracy metrics** — R², MAE, RMSE, MAPE
- Displays **interactive charts** — forecast, residuals, error distribution

> Example: *Given 3 years of monthly sales data → predict next year's sales*

---

## ✨ Features

- ✅ 3 Forecasting Models — Linear Regression, Polynomial Regression, Moving Average
- ✅ 3 Datasets — Monthly Sales Revenue, Temperature Anomaly, User Growth
- ✅ 4 Accuracy Metrics — MAE, RMSE, MAPE, R²
- ✅ Interactive Plotly Charts — forecast + residuals + error distribution
- ✅ Model Comparison — side-by-side metrics table with best-model highlighting
- ✅ CLI Script — `train_and_evaluate.py` for offline batch evaluation

---

## 🗂️ Project Structure

```
predictive-analytics/
├── app.py                    # Streamlit dashboard (main entry point)
├── train_and_evaluate.py     # CLI: train all models, export metrics + plots
├── requirements.txt          # Python dependencies
├── utils/
│   └── data_loader.py        # Synthetic dataset generators
├── models/
│   └── forecaster.py         # Model fitting, prediction, evaluation
└── output/                   # Auto-created: plots + metrics CSV
```

---

## ⚙️ Run Locally

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

| Metric | What it means | Ideal |
|---|---|---|
| MAE | Average absolute error in original units | Lower is better |
| RMSE | Penalizes large errors more heavily | Lower is better |
| MAPE | Error as a percentage | Below 10% is good |
| R² | How well model explains the variance | Closer to 1.0 |

---

## 📈 Sample Results

| Dataset | Best Model | R² | MAPE |
|---|---|---|---|
| Monthly Sales Revenue | Polynomial Regression | 0.9525 | 7.07% |
| Temperature Anomaly | Polynomial Regression | 0.9602 | 6.45% |
| Monthly Active Users | Polynomial Regression | 0.9931 | 8.32% |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.9+ | Core language |
| Streamlit | Interactive web UI |
| Plotly | Interactive charts |
| scikit-learn | Regression models |
| Pandas / NumPy | Data preprocessing |
| Matplotlib | CLI plot exports |

---

## 👤 Author

**Dhivanya**
B.E. Electronics and Communication Engineering
SNS College of Engineering

[![GitHub](https://img.shields.io/badge/GitHub-Dhivanya-black?logo=github)](https://github.com/Dhivanya)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

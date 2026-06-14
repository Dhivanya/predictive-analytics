"""
train_and_evaluate.py — CLI script: trains all models on all datasets,
prints metrics, and saves forecast plots to output/
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from utils.data_loader import load_dataset, DATASETS
from models.forecaster import run_all_models

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "Linear Regression":            "#378ADD",
    "Polynomial Regression (deg=2)":"#639922",
    "Moving Average (window=6)":    "#D85A30",
}

N_FORECAST = 12


def plot_forecast(df: pd.DataFrame, results: dict, dataset_name: str, save_path: str):
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_main = fig.add_subplot(gs[0, :])
    ax_res  = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])

    y      = df["value"].values
    dates  = pd.to_datetime(df["date"])
    future = pd.date_range(start=dates.iloc[-1] + pd.DateOffset(months=1), periods=N_FORECAST, freq="MS")

    ax_main.plot(dates, y, color="#2C2C2A", lw=2, label="Historical", zorder=3)

    all_residuals = {}
    for label, res in results.items():
        color = COLORS.get(label, "#888")
        mask  = ~np.isnan(res["fitted"])
        ax_main.plot(dates[mask], res["fitted"][mask], lw=1.2, ls="--",
                     color=color, alpha=0.85, label=f"{label} (fit)")
        ax_main.plot(future, res["forecast"], lw=2, ls="-.",
                     color=color, marker="D", ms=4, label=f"{label} (forecast)")
        residuals = y[mask] - res["fitted"][mask]
        all_residuals[label] = residuals
        ax_res.scatter(res["fitted"][mask], residuals, alpha=0.6,
                       s=20, color=color, label=label)
        ax_hist.hist(residuals, bins=12, alpha=0.55, color=color, label=label)

    ax_main.axvline(x=dates.iloc[-1], color="gray", lw=1, ls=":", alpha=0.6)
    ax_main.set_title(f"Forecast — {dataset_name}", fontsize=13, fontweight="bold", pad=10)
    ax_main.set_xlabel("Date"); ax_main.set_ylabel(dataset_name)
    ax_main.legend(fontsize=8, ncol=2, loc="upper left")
    ax_main.grid(alpha=0.2)

    ax_res.axhline(0, color="gray", lw=1, ls="--", alpha=0.5)
    ax_res.set_title("Residuals vs Predicted", fontsize=11)
    ax_res.set_xlabel("Predicted"); ax_res.set_ylabel("Residual")
    ax_res.legend(fontsize=7); ax_res.grid(alpha=0.2)

    ax_hist.set_title("Error Distribution", fontsize=11)
    ax_hist.set_xlabel("Residual"); ax_hist.set_ylabel("Count")
    ax_hist.legend(fontsize=7); ax_hist.grid(alpha=0.2)

    plt.suptitle("Predictive Analytics · github.com/Dhivanya", fontsize=9,
                 color="gray", y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def main():
    all_metrics = []

    for ds_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        df      = load_dataset(ds_name)
        y       = df["value"].values
        results = run_all_models(y, N_FORECAST)

        for model_label, res in results.items():
            m = res["metrics"]
            row = {"Dataset": ds_name, "Model": model_label, **m}
            all_metrics.append(row)
            print(f"  [{model_label}]")
            for k, v in m.items():
                print(f"    {k:12s}: {v}")

        safe_name = ds_name.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "").replace("×", "x")
        plot_forecast(df, results, ds_name, os.path.join(OUTPUT_DIR, f"{safe_name}.png"))

    metrics_df = pd.DataFrame(all_metrics)
    csv_path   = os.path.join(OUTPUT_DIR, "model_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n✅ Metrics saved → {csv_path}")
    print("\nTop model per dataset (by R²):")
    print(metrics_df.loc[metrics_df.groupby("Dataset")["R²"].idxmax()][["Dataset", "Model", "R²", "MAPE (%)"]].to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
from math import sqrt
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .baselines import naive, seasonal_naive
from .sarima import fit_predict
from ..utils import ensure_dir

def evaluate(y_true, y_pred) -> tuple[float,float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", choices=["naive","seasonal_naive","sarima"], default="sarima")
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--metrics_out", default="reports/metrics/metrics.csv")
    ap.add_argument("--plot_out", default="reports/figures/test_pred.png")
    args = ap.parse_args()

    df = pd.read_csv(args.dataset, parse_dates=["date"]).set_index("date")
    y = df["tavg"]
    train = df[df["is_test"] == 0]["tavg"]
    test  = df[df["is_test"] == 1]["tavg"]

    if args.model == "naive":
        preds = naive(y).loc[test.index]
    elif args.model == "seasonal_naive":
        preds = seasonal_naive(y, period=365).loc[test.index]
    else:
        preds = fit_predict(train, test, seasonal_periods=7)

    mae, rmse = evaluate(test, preds)

    # write metrics
    mpath = Path(args.metrics_out)
    ensure_dir(mpath.parent)
    import csv
    write_header = not mpath.exists()
    with mpath.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["model","mae","rmse"])
        w.writerow([args.model, f"{mae:.3f}", f"{rmse:.3f}"])

    # plot
    ensure_dir(Path(args.plot_out).parent)
    plt.figure(figsize=(10,4))
    test.plot(label="actual")
    preds.plot(label="pred")
    plt.title(f"{args.model} – test window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=150)
    print(f"{args.model} -> MAE {mae:.3f}  RMSE {rmse:.3f}\nSaved plot to {args.plot_out}")

if __name__ == "__main__":
    main()

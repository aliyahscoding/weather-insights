from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from ..utils import ensure_dir

FEATS = [
    "tavg_lag_1","tavg_lag_7","tavg_lag_14",
    "tavg_roll_mean_7","tavg_roll_std_7","tavg_roll_mean_30",
    "dow","month","is_weekend"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_metrics", default="reports/metrics/metrics.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.dataset, parse_dates=["date"]).set_index("date")
    y = df["tavg"]
    X = df[FEATS]

    train = df[df["is_test"] == 0]
    test  = df[df["is_test"] == 1]

    Xtr, ytr = train[FEATS].dropna(), train["tavg"].loc[train[FEATS].dropna().index]
    Xte, yte = test[FEATS].dropna(),  test["tavg"].loc[test[FEATS].dropna().index]

    model = Ridge(alpha=1.0)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)

    mae = mean_absolute_error(yte, preds)
    rmse = sqrt(mean_squared_error(yte, preds))

    out = Path(args.out_metrics)
    ensure_dir(out.parent)
    # append row
    import csv
    write_header = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["model","mae","rmse"])
        w.writerow(["ridge_ml", f"{mae:.3f}", f"{rmse:.3f}"])
    print(f"ML Ridge -> MAE {mae:.3f}  RMSE {rmse:.3f}")

if __name__ == "__main__":
    main()

"""
Clean raw CSV, build features (lags/rollings/calendar), save processed CSV.

Example:
  python -m src.preprocess --in data/raw/austin_2018_2024.csv --out data/processed/austin_proc.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from .utils import ensure_dir

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df

def add_lags_rolls(df: pd.DataFrame, col: str = "tavg") -> pd.DataFrame:
    for k in [1, 7, 14]:
        df[f"{col}_lag_{k}"] = df[col].shift(k)
    df[f"{col}_roll_mean_7"] = df[col].rolling(7).mean()
    df[f"{col}_roll_std_7"]  = df[col].rolling(7).std()
    df[f"{col}_roll_mean_30"] = df[col].rolling(30).mean()
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.infile, parse_dates=["date"])
    df = df.set_index("date").sort_index()

    # reindex to daily frequency, forward-fill small gaps
    idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(idx)
    # carry forward a few common numeric cols
    for c in ["tmin","tmax","tavg","precip","wind_max","lat","lon"]:
        if c in df.columns:
            df[c] = df[c].interpolate(limit=3)  # small holes, calm down
    # forward fill location string
    if "location" in df.columns:
        df["location"] = df["location"].ffill().bfill()

    df = add_calendar(df)
    df = add_lags_rolls(df, "tavg")
    df.rename_axis("date", inplace=True)

    # Train/test split index marker: last 365 days as test
    df["is_test"] = (df.index >= (df.index.max() - pd.Timedelta(days=364))).astype(int)

    out = Path(args.outfile)
    ensure_dir(out.parent)
    df.to_csv(out, index=True)
    print(f"Processed saved to {out} with shape {df.shape}")

if __name__ == "__main__":
    main()

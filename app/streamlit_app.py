import streamlit as st
from pathlib import Path
import pandas as pd
from datetime import date
from src.utils import slugify, ensure_dir
from src.modeling.baselines import naive, seasonal_naive
from src.modeling.sarima import fit_predict

RAW = Path("data/raw")
PROC = Path("data/processed")

st.set_page_config(page_title="Weather Insights", layout="wide")
st.title("Weather Insights Dashboard")

with st.sidebar:
    st.header("Controls")
    location = st.text_input("Location", value="Austin,US")
    start = st.date_input("Start date", value=date(2018,1,1))
    end   = st.date_input("End date", value=date.today())
    model = st.selectbox("Model", ["seasonal_naive","sarima","naive"])
    horizon = st.slider("Forecast horizon (days)", 7, 30, 14)
    run_ingest = st.button("Fetch/Update Data")

if run_ingest:
    # call CLI scripts via subprocess for simplicity
    import subprocess, sys
    slug = slugify(location)
    raw_out = RAW / f"{slug}_{start.isoformat()}_{end.isoformat()}.csv"
    cmd1 = [sys.executable, "-m", "src.ingest", "--location", location, "--start", start.isoformat(), "--end", end.isoformat(), "--outfile", str(raw_out)]
    st.code(" ".join(cmd1))
    st.info("Ingesting...")
    r1 = subprocess.run(cmd1, capture_output=True, text=True)
    st.text(r1.stdout if r1.stdout else r1.stderr)

    proc_out = PROC / f"{slug}_proc.csv"
    cmd2 = [sys.executable, "-m", "src.preprocess", "--in", str(raw_out), "--out", str(proc_out)]
    st.code(" ".join(cmd2))
    st.info("Preprocessing...")
    r2 = subprocess.run(cmd2, capture_output=True, text=True)
    st.text(r2.stdout if r2.stdout else r2.stderr)

# Try to load any processed file for the chosen location
slug = slugify(location)
proc_path = PROC / f"{slug}_proc.csv"
if proc_path.exists():
    df = pd.read_csv(proc_path, parse_dates=["date"]).set_index("date")
    st.subheader(f"Dataset: {location}")
    st.line_chart(df["tavg"])

    test = df[df["is_test"] == 1]["tavg"]
    y = df["tavg"]
    train = df[df["is_test"] == 0]["tavg"]

    if model == "naive":
        preds = naive(y).loc[test.index]
    elif model == "seasonal_naive":
        preds = seasonal_naive(y).loc[test.index]
    else:
        preds = fit_predict(train, test, seasonal_periods=7)

    # Short horizon forecast: continue last date forward
    last = y.index.max()
    future_idx = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")
    future = preds.reindex(future_idx)  # crude demo; SARIMA supports dynamic forecasts if extended

    st.subheader("Test vs Prediction (last year)")
    chart_df = pd.DataFrame({"actual": test, "pred": preds})
    st.line_chart(chart_df)

    st.subheader(f"{horizon}-day Forecast (toy)")
    st.line_chart(future.dropna())
else:
    st.warning("No processed dataset found yet. Use the sidebar button to fetch data first.")

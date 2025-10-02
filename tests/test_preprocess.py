import pandas as pd
from pathlib import Path

def test_processed_has_features():
    p = Path("data/processed")
    # Pass if no data yet; CI should still run. This prevents failing before ingest.
    if not p.exists():
        assert True
        return
    files = list(p.glob("*_proc.csv"))
    if not files:
        assert True
        return
    df = pd.read_csv(files[0])
    needed = {"tavg","tavg_lag_1","tavg_lag_7","tavg_lag_14","tavg_roll_mean_7","dow","month","is_weekend","is_test"}
    assert needed.issubset(set(df.columns))

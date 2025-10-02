"""
Fetch daily weather using Open-Meteo (no key), save standardized CSV.

Usage example:
  python -m src.ingest --location "Austin,US" --start 2018-01-01 --end 2024-12-31 --outfile data/raw/austin_2018_2024.csv
"""
from __future__ import annotations
import argparse
from datetime import date
from pathlib import Path
import requests
import pandas as pd
from .utils import ensure_dir, slugify, ROOT

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
WX_URL  = "https://api.open-meteo.com/v1/forecast"

def geocode(location: str) -> dict:
    r = requests.get(GEO_URL, params={"name": location, "count": 1, "language": "en", "format": "json"}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data.get("results"):
        raise ValueError(f"Could not geocode location: {location}")
    return data["results"][0]  # dict with latitude, longitude, timezone, country_code, etc.

def fetch_daily(location: str, start: str, end: str) -> pd.DataFrame:
    loc = geocode(location)
    params = {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "start_date": start,
        "end_date": end,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "windspeed_10m_max"
        ]),
        "timezone": "auto",
    }
    r = requests.get(WX_URL, params=params, timeout=60)
    r.raise_for_status()
    dd = r.json().get("daily", {})
    if not dd:
        raise ValueError("No daily data returned.")
    df = pd.DataFrame(dd)
    # Standardize columns
    df.rename(columns={
        "time": "date",
        "temperature_2m_min": "tmin",
        "temperature_2m_max": "tmax",
        "temperature_2m_mean": "tavg",
        "precipitation_sum": "precip",
        "windspeed_10m_max": "wind_max"
    }, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    # Some sources omit mean; if missing, compute it
    if "tavg" not in df or df["tavg"].isna().all():
        df["tavg"] = (df["tmin"] + df["tmax"]) / 2.0
    # Add metadata columns
    df["location"] = f'{loc["name"]},{loc.get("country_code","")}'
    df["lat"] = loc["latitude"]
    df["lon"] = loc["longitude"]
    return df[["date", "tmin", "tmax", "tavg", "precip", "wind_max", "location", "lat", "lon"]]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--location", required=True, help='e.g. "Austin,US"')
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--outfile", default=None, help="Path to CSV in data/raw/")
    args = p.parse_args()

    df = fetch_daily(args.location, args.start, args.end)
    out = Path(args.outfile) if args.outfile else (ROOT / "data" / "raw" / f"{slugify(args.location)}_{args.start}_{args.end}.csv")
    ensure_dir(out.parent)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows to {out}")

if __name__ == "__main__":
    main()

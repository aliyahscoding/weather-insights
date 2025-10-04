"""
Fetch daily weather using Open-Meteo with robust geocoding and coord fallback.
- Accepts either a place string (e.g., "Austin,US" or "Austin, Texas")
  or explicit coords via --coords "30.2672,-97.7431".
- Auto-selects archive vs forecast API based on dates.

Usage (place):
  python -m src.ingest --location "Austin,US" --start 2023-01-01 --end 2023-03-31 --outfile data/raw/austin_q1_2023.csv

Usage (coords to bypass geocoder):
  python -m src.ingest --coords "30.2672,-97.7431" --label "Austin,US" --start 2023-01-01 --end 2023-03-31 --outfile data/raw/austin_q1_2023.csv
"""
from __future__ import annotations
import argparse
from datetime import date, datetime
from pathlib import Path
import re
import requests
import pandas as pd

from .utils import ensure_dir, slugify, ROOT

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "windspeed_10m_max",
]

STATE_MAP = {
    "TX": "Texas", "CA": "California", "NY": "New York", "FL": "Florida",
    "WA": "Washington", "IL": "Illinois", "MA": "Massachusetts"
}

HEADERS = {"User-Agent": "weather-insights/0.1 (educational use)"}

def parse_coords(text: str) -> tuple[float,float] | None:
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", text)
    if not m: return None
    lat = float(m.group(1)); lon = float(m.group(2))
    return lat, lon

def _pick_base_url(start_iso: str, end_iso: str) -> str:
    today = date.today()
    s = datetime.strptime(start_iso, "%Y-%m-%d").date()
    e = datetime.strptime(end_iso, "%Y-%m-%d").date()
    return ARCHIVE_URL if e < today else FORECAST_URL

def geocode_place(place: str) -> dict:
    # Try progressively: exact tokens → city only → city + admin1 guess
    tokens = [t.strip() for t in place.split(",") if t.strip()]
    city = tokens[0] if tokens else place.strip()
    admin1 = None
    country_code = None

    # Infer admin1/country from tokens like "Austin,US" or "Austin,TX"
    if len(tokens) >= 2:
        t1 = tokens[1].upper()
        if len(t1) == 2 and t1.isalpha():
            # Two-letter token: could be US or a state
            if t1 == "US":
                country_code = "US"
            elif t1 in STATE_MAP:
                admin1 = STATE_MAP[t1]; country_code = "US"
        else:
            # Words like "Texas" or "United States"
            if t1.lower() in {v.lower() for v in STATE_MAP.values()}:
                admin1 = t1; country_code = "US"
            if t1.lower() in {"united states", "usa", "u.s.", "us"}:
                country_code = "US"

    # 1) Try city + optional filters
    params = {"name": city, "count": 5, "language": "en", "format": "json"}
    if admin1: params["admin1"] = admin1
    if country_code: params["country_code"] = country_code
    r = requests.get(GEO_URL, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", []) or []

    # If nothing, try city only
    if not results:
        r2 = requests.get(GEO_URL, params={"name": city, "count": 5, "language": "en", "format": "json"},
                          headers=HEADERS, timeout=30)
        r2.raise_for_status()
        data2 = r2.json()
        results = data2.get("results", []) or []

    # If still nothing and we inferred a state name, try forcing admin1 with US
    if not results and admin1:
        r3 = requests.get(GEO_URL, params={"name": city, "admin1": admin1, "country_code": "US", "count": 5,
                                           "language": "en", "format": "json"},
                          headers=HEADERS, timeout=30)
        r3.raise_for_status()
        data3 = r3.json()
        results = data3.get("results", []) or []

    if not results:
        raise ValueError(f"Could not geocode location: {place}")

    # Prefer exact admin1 if we have it, else first result
    if admin1:
        for res in results:
            if str(res.get("admin1","")).lower() == admin1.lower():
                return res
    return results[0]

def fetch_daily_by_coords(lat: float, lon: float, label: str, start: str, end: str) -> pd.DataFrame:
    base = _pick_base_url(start, end)
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": "auto",
    }
    r = requests.get(base, params=params, headers=HEADERS, timeout=90)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        detail = ""
        try: detail = r.json()
        except Exception: detail = r.text[:300]
        raise SystemExit(f"Open-Meteo error {r.status_code} on {r.url}\nDetails: {detail}") from e

    dd = r.json().get("daily", {})
    if not dd:
        raise ValueError(f"No daily data returned from {base}. URL was:\n{r.url}")

    df = pd.DataFrame(dd)
    df.rename(columns={
        "time":"date",
        "temperature_2m_min":"tmin",
        "temperature_2m_max":"tmax",
        "temperature_2m_mean":"tavg",
        "precipitation_sum":"precip",
        "windspeed_10m_max":"wind_max",
    }, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    if "tavg" not in df or df["tavg"].isna().all():
        df["tavg"] = (df["tmin"] + df["tmax"]) / 2.0
    df["location"] = label
    df["lat"] = lat; df["lon"] = lon
    return df[["date","tmin","tmax","tavg","precip","wind_max","location","lat","lon"]]

def fetch_daily(place: str, start: str, end: str) -> pd.DataFrame:
    loc = geocode_place(place)
    label = f'{loc["name"]},{loc.get("country_code","")}'
    return fetch_daily_by_coords(loc["latitude"], loc["longitude"], label, start, end)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--location", help='e.g. "Austin,US" or "Austin, Texas"')
    p.add_argument("--coords", help='lat,lon e.g. "30.2672,-97.7431" (bypasses geocoder)')
    p.add_argument("--label", help='Label for coords mode, e.g. "Austin,US"')
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--outfile", default=None, help="Path to CSV in data/raw/")
    args = p.parse_args()

    if args.coords:
        latlon = parse_coords(args.coords)
        if not latlon:
            raise SystemExit(f"Invalid --coords: {args.coords}. Expected 'lat,lon'.")
        lat, lon = latlon
        label = args.label or f"coords({lat:.4f},{lon:.4f})"
        df = fetch_daily_by_coords(lat, lon, label, args.start, args.end)
    elif args.location:
        df = fetch_daily(args.location, args.start, args.end)
    else:
        raise SystemExit("Provide --location or --coords")

    out = Path(args.outfile) if args.outfile else (ROOT / "data" / "raw" / f"{slugify(df['location'].iloc[0])}_{args.start}_{args.end}.csv")
    ensure_dir(out.parent)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows to {out}")

if __name__ == "__main__":
    main()

# Weather Insights

Small, reproducible pipeline for daily weather: ingest → preprocess → backtest models → Streamlit dashboard.

**Quickstart**
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.ingest --location "Austin,US" --start 2018-01-01 --end 2024-12-31 --outfile data/raw/austin_2018_2024.csv
python -m src.preprocess --in data/raw/austin_2018_2024.csv --out data/processed/austin_proc.csv
python -m src.modeling.backtest --dataset data/processed/austin_proc.csv --model sarima --horizon 7
streamlit run app/streamlit_app.py
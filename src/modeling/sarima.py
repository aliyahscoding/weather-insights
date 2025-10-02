from __future__ import annotations
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_predict(train: pd.Series, test: pd.Series, seasonal_periods: int = 7):
    # fixed small model to avoid rabbit holes
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1, seasonal_periods), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    preds = res.get_prediction(start=test.index.min(), end=test.index.max()).predicted_mean
    return preds

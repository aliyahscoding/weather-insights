from __future__ import annotations
import pandas as pd
import numpy as np

def naive(y: pd.Series) -> pd.Series:
    return y.shift(1)

def seasonal_naive(y: pd.Series, period: int = 365) -> pd.Series:
    ysn = y.shift(period)
    return ysn.fillna(y.shift(1))

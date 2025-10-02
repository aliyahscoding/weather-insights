import pandas as pd
from src.modeling.baselines import naive, seasonal_naive

def test_naive_has_shift():
    s = pd.Series([1,2,3])
    n = naive(s)
    assert pd.isna(n.iloc[0]) and n.iloc[1] == 1

def test_seasonal_falls_back():
    s = pd.Series(range(370))
    sn = seasonal_naive(s, period=365)
    assert sn.iloc[365] == s.iloc[0]
    assert pd.notna(sn.iloc[366])

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .utils import ensure_dir

def line(df: pd.DataFrame, col: str, out: Path):
    ensure_dir(out.parent)
    plt.figure(figsize=(10,4))
    df[col].plot()
    plt.title(col)
    plt.tight_layout()
    plt.savefig(out, dpi=150)

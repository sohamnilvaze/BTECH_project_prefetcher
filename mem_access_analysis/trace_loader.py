'''
reads the .csv file, ensures that numeric columns are float, fills the missing values with 0.0 
'''


import pandas as pd
from pathlib import Path

def load_trace_csv(path_or_pattern):
    """Loads a CSV or all CSVs matching a glob; returns dict filename->df"""
    p = Path(path_or_pattern)
    if p.is_file():
        df = pd.read_csv(p)
        return {p.stem: df}
    else:
        dfs = {}
        for f in sorted(Path('.').glob(path_or_pattern)):
            dfs[f.stem] = pd.read_csv(f)
        return dfs

def sanitize_df(df, feature_cols):
    # ensure features exist and are numeric
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].astype(float).fillna(0.0)
    return df

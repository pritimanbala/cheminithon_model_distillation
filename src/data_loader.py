from pathlib import Path
import pandas as pd
REQUIRED_FEED=['feed_MeOH','feed_EtAC','feed_Water','feed_flow','feed_pressure','feed_temperature']
def load_plant_data(path,column_mapping=None,timestamp_column=None):
    path=Path(path); df=pd.read_excel(path) if path.suffix.lower() in ('.xlsx','.xls') else pd.read_csv(path)
    if column_mapping: df=df.rename(columns=column_mapping)
    missing=[c for c in REQUIRED_FEED if c not in df]
    if missing: raise ValueError(f"Missing required mapped columns: {missing}")
    if df[REQUIRED_FEED].isna().any().any(): raise ValueError('Missing required values; imputation requires explicit configuration')
    if df.duplicated().any(): raise ValueError('Duplicate records detected')
    if ((df[['feed_MeOH','feed_EtAC','feed_Water']]<0).any().any() or ((df[['feed_MeOH','feed_EtAC','feed_Water']].sum(axis=1)-1).abs()>1e-4).any()): raise ValueError('Invalid feed composition sums')
    if timestamp_column:
        if timestamp_column not in df: raise ValueError('Configured timestamp column absent')
        df[timestamp_column]=pd.to_datetime(df[timestamp_column],errors='raise'); df=df.sort_values(timestamp_column)
    return df

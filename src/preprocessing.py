"""Leakage-safe chronological splitting utilities for real plant observations."""
def chronological_split(df, train_fraction=.7, validation_fraction=.15):
    n=len(df); a=int(n*train_fraction); b=int(n*(train_fraction+validation_fraction))
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()

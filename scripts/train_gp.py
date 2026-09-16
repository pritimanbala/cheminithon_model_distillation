import sys; sys.path.insert(0,'.')
import pandas as pd,yaml
from src.gp_classifier import ParetoGPClassifier
cfg=yaml.safe_load(open('config/config.yaml')); df=pd.read_csv('data/synthetic/fced_synthetic_5000.csv').dropna(); X=df[list(cfg['variables'])]; y=(df['feasibility_class']=='B').astype(int); g=ParetoGPClassifier(); g.fit(X,y); print(g.metrics)

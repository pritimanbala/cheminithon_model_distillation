import sys; sys.path.insert(0,'.')
from pathlib import Path
import yaml
from src.synthetic_data import generate
cfg=yaml.safe_load(open('config/config.yaml')); df=generate(5000,cfg['variables'],cfg['seed']); Path('data/synthetic').mkdir(parents=True,exist_ok=True); df.to_csv('data/synthetic/fced_synthetic_5000.csv',index=False); print(f'Wrote {len(df)} synthetic records')

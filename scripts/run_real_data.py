import sys; sys.path.insert(0,'.')
import argparse
from src.data_loader import load_plant_data
p=argparse.ArgumentParser(description='Validate real plant data; does not replace it with synthetic data.'); p.add_argument('--file',required=True); p.add_argument('--mode',default='adaptive_operational'); p.add_argument('--timestamp-column'); a=p.parse_args()
df=load_plant_data(a.file,timestamp_column=a.timestamp_column); print(f'Validated {len(df)} real records for {a.mode}; train a validated model before recommendations.')

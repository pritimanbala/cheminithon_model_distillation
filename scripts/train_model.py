import sys; sys.path.insert(0,'.')
import json,pandas as pd
from src.performance_model import EnsemblePerformanceModel
from src.model_registry import ModelRegistry
df=pd.read_csv('data/synthetic/fced_synthetic_5000.csv').dropna(); features=['feed_MeOH','feed_EtAC','feed_Water','feed_flow','feed_pressure','feed_temperature','reflux_ratio','reboiler_duty','condenser_duty','feed_stage_location','column_pressure','feed_flow_rate']; targets=['purity_MeOH','purity_EtAC','recovery_MeOH','recovery_EtAC','energy_consumption','throughput','operating_cost']
m=EnsemblePerformanceModel().fit(df[features],df[targets]); fn=ModelRegistry().save(m,{'dataset_version':'synthetic_5000','features':features,'model_type':'RandomForestRegressor','training_sample_count':len(df),'validation_metrics':'not calculated in prototype script','synthetic':True}); print('Saved',fn)

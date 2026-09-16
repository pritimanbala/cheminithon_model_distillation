import sys; sys.path.insert(0,'.')
import yaml
from src.modidc import MODIDCOptimizer
from src.process_simulator import SyntheticFCEDSimulator
from src.constraints import ConstraintSet
from src.objectives import operational
cfg=yaml.safe_load(open('config/config.yaml')); feed={'feed_MeOH':.05,'feed_EtAC':.05,'feed_Water':.90,'feed_flow':100,'feed_pressure':1,'feed_temperature':25}
r=MODIDCOptimizer(cfg['variables'],cfg['population_size'],cfg['generations'],seed=cfg['seed']).optimize(SyntheticFCEDSimulator(),feed,ConstraintSet(cfg['variables'],cfg['purity_targets'],cfg['recovery_targets']),operational,use_correction=False); print('Standard NSGA-II feasible Pareto:',len(r['pareto']))

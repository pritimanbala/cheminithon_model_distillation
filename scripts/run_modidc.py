import sys; sys.path.insert(0,'.')
from pathlib import Path
import json,yaml,pandas as pd
from src.process_simulator import SyntheticFCEDSimulator
from src.constraints import ConstraintSet
from src.objectives import operational,paper
from src.modidc import MODIDCOptimizer
from src.topsis import select
from src.evaluation import summary
from src.visualization import plots
import matplotlib.pyplot as plt
cfg=yaml.safe_load(open('config/config.yaml')); Path('outputs').mkdir(exist_ok=True)
feed={'feed_MeOH':.05,'feed_EtAC':.05,'feed_Water':.90,'feed_flow':100,'feed_pressure':1.0,'feed_temperature':25}
specs=cfg['paper_variables'] if cfg['mode']=='paper_replication' else cfg['variables']; con=ConstraintSet(specs,cfg['purity_targets'],cfg['recovery_targets']); obj=paper if cfg['mode']=='paper_replication' else operational
r=MODIDCOptimizer(specs,cfg['population_size'],cfg['generations'],cfg['gp_probability_threshold'],cfg['seed']).optimize(SyntheticFCEDSimulator(),feed,con,obj)
rows=[{**e['x'],**e['y']['purity'],**{f'recovery_{k}':v for k,v in e['y']['recovery'].items()},**{f'objective_{i}':v for i,v in enumerate(e['objectives'])}} for e in r['pareto']]; pd.DataFrame(rows).to_csv('outputs/final_pareto.csv',index=False); pd.DataFrame(r['history']).to_csv('outputs/generation_history.csv',index=False); pd.DataFrame(r['corrections']).to_csv('outputs/correction_matrix.csv',index=False); pd.DataFrame([r['gp_metrics']]).to_csv('outputs/gp_metrics.csv',index=False)
if r['pareto']:
 i,close=select([e['objectives'] for e in r['pareto']], list(cfg['operating_weights'].values()) if cfg['mode']!='paper_replication' else None); chosen=r['pareto'][i]; pd.DataFrame([{**chosen['x'],**chosen['y'],'topsis_closeness':close[i]}]).to_csv('outputs/balanced_solution.csv',index=False); rec={'status':'REQUIRES VALIDATION','label':'TOPSIS-selected balanced Pareto operating point','synthetic':True,'decision_variables':chosen['x'],'model_confidence_note':'Synthetic simulator only; operator approval required'}; Path('outputs/recommendation.json').write_text(json.dumps(rec,indent=2,default=str)); plots(r['pareto'])
Path('outputs/optimization_summary.json').write_text(json.dumps(summary(r),indent=2)); Path('outputs/model_validation_report.json').write_text(json.dumps({'synthetic':True,'status':'NOT A PLANT-VALIDATED MODEL','gp_classifier_metrics':r['gp_metrics']},indent=2,default=str))
hist=pd.DataFrame(r['history']); plt.figure(); plt.plot(hist['generation'],hist['feasible']); plt.xlabel('Generation'); plt.ylabel('Feasible candidates'); plt.title('Synthetic convergence'); plt.tight_layout(); plt.savefig('outputs/convergence.png'); plt.close(); print('Completed synthetic MO-DIDC run.')

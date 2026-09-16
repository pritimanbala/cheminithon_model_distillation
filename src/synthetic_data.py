import numpy as np, pandas as pd
from .process_simulator import SyntheticFCEDSimulator
from .constraints import ConstraintSet
def generate(n, specs, seed=42):
    rng=np.random.default_rng(seed); sim=SyntheticFCEDSimulator(); c=ConstraintSet(specs,{'MeOH':.95,'EtAC':.95},{'MeOH':.85,'EtAC':.85}); rows=[]
    design_specs={'N_TPC':(4,12),'N_FPC':(2,10),'N_TEDC':(20,55),'N_FEDC':(4,50),'N_EEDC':(2,30),'F_EDC':(5,50),'RR_EDC':(.1,4),'N_TPDC':(10,35),'N_FPDC':(2,30),'RR_PDC':(.1,6),'N_TSRC':(8,28),'N_FSRC':(2,24),'RR_SRC':(.05,3)}
    for i in range(n):
        water=rng.uniform(.7,.96); meoh=rng.uniform(.01,1-water-.005); etac=1-water-meoh; feed={'feed_MeOH':meoh,'feed_EtAC':etac,'feed_Water':water,'feed_flow':rng.uniform(50,150),'feed_pressure':rng.uniform(.8,2),'feed_temperature':rng.uniform(20,80)}
        x={k:(int(rng.integers(s['lower'],s['upper']+1)) if s.get('kind')=='integer' else float(rng.uniform(s['lower'],s['upper']))) for k,s in specs.items()}
        d={k:(int(rng.integers(a,b+1)) if k.startswith('N_') else float(rng.uniform(a,b))) for k,(a,b) in design_specs.items()}; d['N_FPC']=min(d['N_FPC'],d['N_TPC']-1); d['N_FEDC']=min(d['N_FEDC'],d['N_TEDC']-1); d['N_EEDC']=min(d['N_EEDC'],d['N_FEDC']); d['N_FPDC']=min(d['N_FPDC'],d['N_TPDC']-1); d['N_FSRC']=min(d['N_FSRC'],d['N_TSRC']-1)
        try:
            y=sim.evaluate(x,{**feed,'F_EDC':d['F_EDC']}); valid,errs=c.validate(x,y); row={**feed,**x,**d,'purity_MeOH':y['purity']['MeOH'],'purity_EtAC':y['purity']['EtAC'],'recovery_MeOH':y['recovery']['MeOH'],'recovery_EtAC':y['recovery']['EtAC'],**{k:y[k] for k in ('energy_consumption','throughput','operating_cost','TAC','GGE_CO2','PRI')},'variable_constraints_valid':True,'simulation_converged':True,'feasibility_class':'B' if valid else 'A','synthetic':True}
        except Exception:
            row={**feed,**x,**d,'variable_constraints_valid':True,'simulation_converged':False,'feasibility_class':'C','synthetic':True}
        rows.append(row)
    return pd.DataFrame(rows)

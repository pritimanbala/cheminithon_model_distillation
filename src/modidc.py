import numpy as np
from .nsga2 import NSGA2
from .pareto import nondominated_mask
from .gp_classifier import ParetoGPClassifier
from .directed_correction import correct
class MODIDCOptimizer:
    def __init__(self,specs,population_size=48,generations=12,probability_threshold=.5,seed=42):
        self.specs=specs; self.n=population_size; self.gens=generations; self.threshold=probability_threshold; self.rng=np.random.default_rng(seed)
    def optimize(self,process_model,feed_conditions,constraints,objective_definition,warm_start=None,use_correction=True):
        keys=list(self.specs); gp=ParetoGPClassifier(); corrections=[]; stats={'promising_infeasible':0}
        def evaluate(x):
            ok,errs=constraints.validate_variables(x)
            if not ok: return {'x':x,'feasible':False,'class':'A','errors':errs,'objectives':np.full(5,1e12),'y':{}}
            try: y=process_model.evaluate(x,feed_conditions)
            except Exception as exc: return {'x':x,'feasible':False,'class':'C','errors':[str(exc)],'objectives':np.full(5,1e12),'y':{}}
            feasible,errs=constraints.validate(x,y); return {'x':x,'y':y,'feasible':feasible,'class':'B' if feasible else 'A','errors':errs,'objectives':objective_definition(y,constraints)}
        engine=NSGA2(self.specs,self.n,self.gens,int(self.rng.integers(2**31)))
        pop=(warm_start or [])[:self.n]+[engine.sample() for _ in range(self.n-len((warm_start or [])[:self.n]))]
        def hook(evaluated,elites,g):
            feasible=[e for e in evaluated if e['feasible']]; class_c=[e for e in evaluated if e['class']=='C']; additions=[]
            if feasible:
                F=np.array([e['objectives'] for e in feasible]); pm=nondominated_mask(F); labels=pm.astype(int); X=np.array([[e['x'][k] for k in keys] for e in feasible])
                gp.fit(X,labels)
                if use_correction and gp.model is not None and class_c:
                    cx=np.array([[e['x'][k] for k in keys] for e in class_c]); probs=gp.probability(cx); chosen=[e['x'] for e,p in zip(class_c,probs) if p>=self.threshold]; stats['promising_infeasible']+=len(chosen)
                    pareto=[e['x'] for e,b in zip(feasible,pm) if b]; new,rows=correct(chosen,[e['x'] for e in feasible],pareto,self.specs,self.rng); additions.extend(new); [r.update({'generation':g}) for r in rows]; corrections.extend(rows)
            base=elites+additions
            return base[:self.n]+[engine.mutate(base[self.rng.integers(len(base))]) for _ in range(self.n-len(base))]
        final_pop,history=engine.evolve(pop,evaluate,hook)
        final=[evaluate(x) for x in final_pop]; feasible=[e for e in final if e['feasible']]
        if feasible:
            mask=nondominated_mask(np.array([e['objectives'] for e in feasible])); pareto=[e for e,b in zip(feasible,mask) if b]
        else: pareto=[]
        return {'pareto':pareto,'population':final,'history':history,'corrections':corrections,'gp_metrics':gp.metrics,'stats':stats}

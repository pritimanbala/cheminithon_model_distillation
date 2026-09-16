import numpy as np
from .pareto import ranks
class NSGA2:
    def __init__(self,specs,population_size=48,generations=12,seed=42): self.specs,self.n,self.gens,self.rng=specs,population_size,generations,np.random.default_rng(seed)
    def sample(self):
        x={}
        for k,s in self.specs.items():
            v=self.rng.uniform(s['lower'],s['upper']); x[k]=int(round(v)) if s.get('kind')=='integer' else float(v)
        return x
    def mutate(self,x):
        y=dict(x)
        for k,s in self.specs.items():
            if self.rng.random()<.25:
                v=y[k]+self.rng.normal(0,.12*(s['upper']-s['lower'])); y[k]=float(np.clip(round(v) if s.get('kind')=='integer' else v,s['lower'],s['upper']))
        return y
    def evolve(self,pop,evaluate,on_generation=None):
        history=[]
        for g in range(self.gens):
            evaluated=[evaluate(x) for x in pop]; valid=[e for e in evaluated if e['feasible']]
            # feasibility first, then non-dominated rank; preserves constrained optimization.
            F=np.array([e['objectives'] for e in valid]) if valid else np.empty((0,1)); rr=ranks(F) if len(valid) else []
            for e,r in zip(valid,rr): e['rank']=int(r)
            ordered=sorted(evaluated,key=lambda e:(not e['feasible'],e.get('rank',999),sum(e['objectives'])))
            elites=[e['x'] for e in ordered[:max(2,self.n//2)]]
            history.append({'generation':g,'feasible':len(valid),'class_A':sum(e['class']=='A' for e in evaluated),'class_C':sum(e['class']=='C' for e in evaluated),'best_energy':min((e['y']['energy_consumption'] for e in valid),default=None)})
            if on_generation: pop=on_generation(evaluated,elites,g)
            else: pop=elites+[self.mutate(elites[self.rng.integers(len(elites))]) for _ in range(self.n-len(elites))]
        return pop,history

import numpy as np
def summary(result):
 p=result['pareto']; return {'synthetic':True,'feasible_pareto_solutions':len(p),'minimum_energy':min((x['y']['energy_consumption'] for x in p),default=None),'maximum_throughput':max((x['y']['throughput'] for x in p),default=None),'minimum_operating_cost':min((x['y']['operating_cost'] for x in p),default=None),'class_A':sum(h['class_A'] for h in result['history']),'class_C':sum(h['class_C'] for h in result['history']),'promising_infeasible':result['stats']['promising_infeasible'],'directed_corrections':len(result['corrections'])}

import numpy as np
def operational(y, constraints):
    p_loss = np.mean([max(0,t-y["purity"].get(c,0)) for c,t in constraints.purity_targets.items()])
    r_loss = np.mean([max(0,t-y["recovery"].get(c,0)) for c,t in constraints.recovery_targets.items()])
    return np.array([y["energy_consumption"], p_loss, r_loss, -y["throughput"], y["operating_cost"]],float)
def paper(y, constraints=None): return np.array([y["TAC"],y["GGE_CO2"],y["PRI"]],float)

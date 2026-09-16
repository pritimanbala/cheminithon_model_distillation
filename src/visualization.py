from pathlib import Path
import matplotlib.pyplot as plt
def plots(pareto,out='outputs'):
 Path(out).mkdir(exist_ok=True)
 if not pareto:return
 y=[x['y'] for x in pareto]; e=[z['energy_consumption'] for z in y]; t=[z['throughput'] for z in y]; c=[z['operating_cost'] for z in y]; p=[(z['purity']['MeOH']+z['purity']['EtAC'])/2 for z in y]
 for name,xv,yv,xl,yl in [('pareto_2d.png',e,p,'Energy','Mean purity'),('throughput_energy.png',t,e,'Throughput','Energy'),('energy_cost.png',e,c,'Energy','Cost')]:
  plt.figure();plt.scatter(xv,yv);plt.xlabel(xl);plt.ylabel(yl);plt.title('Synthetic feasible Pareto front');plt.tight_layout();plt.savefig(Path(out)/name);plt.close()

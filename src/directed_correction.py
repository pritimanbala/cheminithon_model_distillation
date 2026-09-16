import numpy as np
def correct(candidates, feasible, pareto, specs, rng):
    """Paper-style variable-wise normalized distance correction with auditable matrix."""
    keys=list(specs); rows=[]; corrected=[]
    if not len(feasible) or not len(pareto): return corrected,rows
    lo=np.array([specs[k]["lower"] for k in keys]); hi=np.array([specs[k]["upper"] for k in keys]); scale=hi-lo
    f=(np.array([[z[k] for k in keys] for z in feasible])-lo)/scale
    for cid,x in enumerate(candidates):
        z=(np.array([x[k] for k in keys])-lo)/scale; dist=np.abs(f-z).mean(0); idx=int(dist.argmax()); key=keys[idx]; lp=min(p[key] for p in pareto); up=max(p[key] for p in pareto); r=float(rng.random()); val=lp+r*(up-lp)
        if specs[key].get("kind")=="integer": val=round(val)
        val=float(np.clip(val,specs[key]["lower"],specs[key]["upper"])); new=dict(x); new[key]=val; corrected.append(new)
        rows.append({"candidate_id":cid,"selected_variable":key,"normalized_candidate_value":z[idx],"average_distance":dist[idx],"LP":lp,"UP":up,"r":r,"corrected_value":val,"original_value":x[key],"correction_magnitude":abs(val-x[key])})
    return corrected,rows

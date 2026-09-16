import numpy as np
def dominates(a,b): return np.all(a<=b) and np.any(a<b)
def nondominated_mask(F):
    return np.array([not any(dominates(F[j],F[i]) for j in range(len(F)) if j!=i) for i in range(len(F))])
def ranks(F):
    remaining=set(range(len(F))); out=np.full(len(F),999); rank=0
    while remaining:
        ids=list(remaining); m=nondominated_mask(F[ids]); front=[ids[i] for i,v in enumerate(m) if v]
        for i in front: out[i]=rank; remaining.remove(i)
        rank+=1
    return out

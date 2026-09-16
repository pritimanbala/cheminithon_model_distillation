import numpy as np
def select(F, weights=None):
    F=np.asarray(F,float); weights=np.ones(F.shape[1])/F.shape[1] if weights is None else np.asarray(weights,float)/sum(weights)
    z=F/(np.sqrt((F**2).sum(axis=0))+1e-12)*weights
    ideal=z.min(0); anti=z.max(0); dplus=np.linalg.norm(z-ideal,axis=1); dminus=np.linalg.norm(z-anti,axis=1)
    c=dminus/(dplus+dminus+1e-12); return int(c.argmax()),c

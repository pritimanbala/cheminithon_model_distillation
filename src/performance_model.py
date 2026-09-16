from abc import ABC,abstractmethod
import numpy as np
from sklearn.ensemble import RandomForestRegressor
class PerformanceModel(ABC):
 @abstractmethod
 def fit(self,X,y): ...
 @abstractmethod
 def predict(self,X): ...
 @abstractmethod
 def predict_with_uncertainty(self,X): ...
class EnsemblePerformanceModel(PerformanceModel):
 def __init__(self,**kwargs): self.model=RandomForestRegressor(n_estimators=150,random_state=42,**kwargs)
 def fit(self,X,y): self.model.fit(X,y); return self
 def predict(self,X): return self.model.predict(X)
 def predict_with_uncertainty(self,X):
  p=np.array([t.predict(X) for t in self.model.estimators_]); return p.mean(0),p.std(0)

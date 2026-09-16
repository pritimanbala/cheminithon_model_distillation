import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
class ParetoGPClassifier:
    def __init__(self): self.scaler=StandardScaler(); self.model=None; self.metrics={}
    def fit(self,X,y):
        y=np.asarray(y)
        if len(y)<8 or len(np.unique(y))<2: self.metrics={"skipped":"insufficient positive/negative samples"}; return False
        Z=self.scaler.fit_transform(X); self.model=GaussianProcessClassifier(kernel=RBF(), random_state=42).fit(Z,y)
        p=self.model.predict_proba(Z)[:,1]; pred=p>=.5; pr,rc,f,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
        self.metrics={"accuracy":accuracy_score(y,pred),"precision":pr,"recall":rc,"f1":f,"roc_auc":roc_auc_score(y,p)}; return True
    def probability(self,X): return np.zeros(len(X)) if self.model is None else self.model.predict_proba(self.scaler.transform(X))[:,1]

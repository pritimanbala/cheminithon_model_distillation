import numpy as np
from src.pareto import nondominated_mask
from src.topsis import select
from src.constraints import ConstraintSet
def test_pareto(): assert nondominated_mask(np.array([[1,2],[2,1],[3,3]])).tolist()==[True,True,False]
def test_topsis(): assert select([[1,1],[2,2]])[0]==0
def test_constraints():
 c=ConstraintSet({'x':{'lower':0,'upper':1,'kind':'continuous'}},{'A':.9},{'A':.8}); assert c.validate({'x':.5},{'purity':{'A':.95},'recovery':{'A':.9}})[0]

class ConstraintSet:
    def __init__(self, variables, purity_targets=None, recovery_targets=None):
        self.variables, self.purity_targets, self.recovery_targets = variables, purity_targets or {}, recovery_targets or {}
    def validate_variables(self, x):
        errors=[]
        for key, spec in self.variables.items():
            if key not in x or not spec["lower"] <= x[key] <= spec["upper"]: errors.append(f"{key} outside configured bounds")
            if key in x and spec.get("kind") == "integer" and int(x[key]) != x[key]: errors.append(f"{key} must be integer")
        if 'N_TPC' in x and not x['N_TPC'] > x['N_FPC']: errors.append('N_TPC must exceed N_FPC')
        if 'N_TEDC' in x and not (x['N_TEDC'] > x['N_FEDC'] >= x['N_EEDC']): errors.append('EDC stage ordering invalid')
        if 'N_TPDC' in x and not x['N_TPDC'] > x['N_FPDC']: errors.append('N_TPDC must exceed N_FPDC')
        if 'N_TSRC' in x and not x['N_TSRC'] > x['N_FSRC']: errors.append('N_TSRC must exceed N_FSRC')
        return not errors, errors
    def validate_performance(self, y):
        errors=[]
        for c,t in self.purity_targets.items():
            if y["purity"].get(c, -1) < t: errors.append(f"{c} purity below target")
        for c,t in self.recovery_targets.items():
            if y["recovery"].get(c, -1) < t: errors.append(f"{c} recovery below target")
        return not errors, errors
    def validate(self,x,y):
        a,e=self.validate_variables(x)
        b,f=self.validate_performance(y) if a else (False,[])
        return a and b, e+f

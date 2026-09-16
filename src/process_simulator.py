from abc import ABC, abstractmethod
import math

class ProcessSimulator(ABC):
    @abstractmethod
    def evaluate(self, decision_variables, feed_conditions): ...

class SyntheticFCEDSimulator(ProcessSimulator):
    """Qualitative development surrogate, not a rigorous thermodynamic/Aspen model."""
    components = ("MeOH", "EtAC")
    def evaluate(self, x, feed):
        # Map paper FCED design decision vector into synthetic proxy operating conditions.
        rr=x.get("reflux_ratio", x.get("RR_EDC", 1.0)); reb=x.get("reboiler_duty", 900+45*x.get("N_TEDC",25)); cond=x.get("condenser_duty",700+30*x.get("N_TPDC",18)); stage=x.get("feed_stage_location",x.get("N_FEDC",15)); pressure=x.get("column_pressure",1.2); flow=x.get("feed_flow_rate",feed.get("feed_flow",100))
        comp = feed["feed_MeOH"], feed["feed_EtAC"], feed["feed_Water"]
        if min(comp) < 0 or abs(sum(comp)-1) > 1e-5: raise ValueError("invalid feed composition")
        # Combination-driven convergence failures: hydraulic/thermal loading and bad stage/reflux combinations.
        if (flow > 155 and rr < 0.7) or (pressure > 2.65 and reb < 750) or (stage < 6 and flow > 130) or (rr > 5.6 and cond < 500):
            raise RuntimeError("synthetic convergence failure")
        sep = (1-math.exp(-0.72*rr)) * (1-math.exp(-reb/980)) * (1-math.exp(-cond/800))
        stage_factor = math.exp(-((stage-18)/15)**2) * .18 + .82
        pressure_factor = 1 - .10*abs(pressure-1.25)
        load = max(.55, 1 - .0025*(flow-100))
        entrainer = x.get("F_EDC", feed.get("F_EDC", 22.0))
        entrainer_factor = .88 + .12*math.exp(-((entrainer-22)/15)**2)
        quality = max(0, min(0.9995, sep*stage_factor*pressure_factor*load*entrainer_factor))
        # High-reflux/high-duty regions can meet prototype specifications; water load still degrades quality.
        purity = {"MeOH": max(.01, min(.9999, .76+.25*quality-.04*comp[2])), "EtAC": max(.01, min(.9999, .74+.27*quality-.03*comp[2]))}
        recovery = {"MeOH": max(.01,min(.999,.40+.60*quality-.0005*max(flow-100,0))), "EtAC": max(.01,min(.999,.38+.62*quality-.0006*max(flow-100,0)))}
        energy = reb + cond + 90*rr + 35*pressure + .8*flow
        cost = energy*.035 + flow*.25 + 5*rr**2
        return {"purity": purity, "recovery": recovery, "energy_consumption": energy, "throughput": flow*quality, "operating_cost": cost, "reboiler_duty":reb, "condenser_duty":cond,
                "TAC": cost*8760, "GGE_CO2": energy*.00018, "PRI": 1/(quality+.05), "synthetic": True}

class AspenFCEDSimulator(ProcessSimulator):
    def evaluate(self, decision_variables, feed_conditions):
        raise NotImplementedError("Connect validated Aspen interface here; optimizer API remains unchanged.")

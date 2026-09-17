"""Adapter: lets the existing MODIDCOptimizer query a trained surrogate."""
from __future__ import annotations

import joblib
import pandas as pd


class ModelBackedFCEDSimulator:
    def __init__(self, artifact_path: str, min_convergence_probability: float = 0.80):
        self.artifact = joblib.load(artifact_path)
        self.features = self.artifact["features"]
        self.targets = self.artifact["targets"]
        self.regressor = self.artifact["outcome_regressor"]
        self.classifier = self.artifact["convergence_classifier"]
        self.minimum_probability = min_convergence_probability

    def evaluate(self, decision_variables: dict, feed_conditions: dict) -> dict:
        row = {**feed_conditions, **decision_variables}
        missing = [name for name in self.features if name not in row]
        if missing:
            raise ValueError(f"Missing model features: {missing}")
        X = pd.DataFrame([[row[name] for name in self.features]], columns=self.features)
        if self.classifier is not None:
            probability = float(self.classifier.predict_proba(X)[0, 1])
            if probability < self.minimum_probability:
                # MODIDC records this as Class C and can attempt directed repair.
                raise RuntimeError(f"predicted convergence probability {probability:.3f}")
        else:
            probability = None
        values = dict(zip(self.targets, self.regressor.predict(X)[0]))
        return {
            "purity": {"MeOH": values["purity_MeOH"], "EtAC": values["purity_EtAC"]},
            "recovery": {"MeOH": values["recovery_MeOH"], "EtAC": values["recovery_EtAC"]},
            "energy_consumption": values["energy_consumption"],
            "throughput": values["throughput"],
            "operating_cost": values["operating_cost"],
            "convergence_probability": probability,
            "synthetic": False,
        }

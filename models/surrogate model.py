"""Train the performance model used by an FCED optimizer.

The model predicts plant/process outcomes from feed context and candidate
operating conditions.  It does NOT predict a single operating set point.
NSGA-II should search this model's predictions subject to hard constraints.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score


DEFAULT_FEATURES = [
    "feed_MeOH", "feed_EtAC", "feed_Water", "feed_flow",
    "feed_pressure", "feed_temperature", "reflux_ratio", "reboiler_duty",
    "condenser_duty", "feed_stage_location", "column_pressure",
    "feed_flow_rate",
]
DEFAULT_TARGETS = [
    "purity_MeOH", "purity_EtAC", "recovery_MeOH", "recovery_EtAC",
    "energy_consumption", "throughput", "operating_cost",
]


def chronological_split(frame: pd.DataFrame, timestamp: str | None):
    """Return 70/15/15 train/validation/test splits without future leakage."""
    if timestamp:
        frame = frame.sort_values(timestamp)
    n = len(frame)
    a, b = int(0.70 * n), int(0.85 * n)
    return frame.iloc[:a], frame.iloc[a:b], frame.iloc[b:]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="CSV of historical or simulator records")
    p.add_argument("--output", default="models/fced_surrogate.joblib")
    p.add_argument("--timestamp", default=None, help="Use this column for chronological split")
    p.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    p.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--converged-column", default="converged")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    needed = set(args.features + args.targets)
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if args.timestamp and args.timestamp not in df:
        raise ValueError(f"Timestamp column not found: {args.timestamp}")

    df = df.dropna(subset=args.features + args.targets).copy()
    # Use recorded convergence if available. Otherwise all complete records are
    # assumed evaluable; define a real plant-specific feasibility label later.
    if args.converged_column in df:
        # CSVs commonly encode this as True/False, 1/0, or strings.
        df["_converged_target"] = df[args.converged_column].astype(str).str.lower().isin(
            ["true", "1", "yes", "converged", "success", "ok"]
        )
    else:
        df["_converged_target"] = True
    if len(df) < 100:
        raise ValueError("Need at least 100 complete records before training.")

    train, valid, test = chronological_split(df, args.timestamp)
    X_train, X_valid, X_test = train[args.features], valid[args.features], test[args.features]
    # Classifier prevents the optimizer from trusting regions likely to fail.
    classifier = None
    if train["_converged_target"].nunique() == 2:
        classifier = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ).fit(X_train, train["_converged_target"])

    # Regress outcomes only where the process produced usable measurements.
    train_ok = train[train["_converged_target"]]
    if len(train_ok) < 50:
        raise ValueError("Too few converged training records for outcome regression.")
    regressor = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=2, max_features=0.8,
        random_state=42, n_jobs=-1,
    ).fit(train_ok[args.features], train_ok[args.targets])

    def report(name, split):
        actual = split[args.targets].to_numpy()
        predicted = regressor.predict(split[args.features])
        return {
            "split": name,
            "mae": dict(zip(args.targets, mean_absolute_error(actual, predicted, multioutput="raw_values").round(5))),
            "r2": dict(zip(args.targets, r2_score(actual, predicted, multioutput="raw_values").round(5))),
        }

    metrics = [report("validation", valid), report("test", test)]
    if classifier is not None and test["_converged_target"].nunique() == 2:
        metrics.append({"test_convergence_roc_auc": round(roc_auc_score(
            test["_converged_target"], classifier.predict_proba(X_test)[:, 1]), 5)})
    artifact = {
        "features": args.features, "targets": args.targets,
        "outcome_regressor": regressor, "convergence_classifier": classifier,
        "metrics": metrics,
        "warning": "Validate against an independent, chronological plant holdout before production use.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    print(pd.Series(metrics, dtype=object).to_string())
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
# AI-assisted multicomponent distillation optimization

## Purpose

This repository is an AI-assisted **decision-support** prototype for multicomponent distillation. It studies candidate operating conditions, searches for feasible energy-quality-throughput trade-offs, and presents results for an operator or process engineer to review. It never writes to a PLC, DCS, Aspen, or plant control system.

The reference architecture is four-column extractive distillation (FCED): PC (preconcentration column), EDC (extractive distillation column), PDC (product distillation column), and SRC (solvent recovery column). The development reference feed is 100 kmol/h with 0.05 MeOH, 0.05 EtAC, and 0.90 water mole fractions; DMSO is represented as the entrainer.

> All current datasets, equations, models, outputs, and recommendations are **synthetic**. They demonstrate software architecture only. They are not rigorous thermodynamic equations, Aspen results, or validated plant predictions.

## What is optimized

For a fixed feed context, the primary operational mode searches configurable decision variables including reflux ratio, reboiler duty, condenser duty, feed-stage location, column pressure, and feed flow rate. It does not collapse competing goals into an arbitrary single score during evolutionary search. It minimizes:

```text
[energy consumption, purity loss, recovery loss, -throughput, operating cost]
```

Purity and recovery are both objectives and hard constraints. A point with low energy but insufficient purity/recovery is not a valid recommendation. Targets, limits, variable types, population size, generations, and TOPSIS weights are configured in [config/config.yaml](config/config.yaml). The included values are prototype defaults, not approved plant limits.

## Operational mode versus paper mode

`adaptive_operational` is the primary project mode. Feed composition, feed flow, pressure, and temperature are context variables; the goal is to adapt operating recommendations as that context changes.

`paper_replication` is a separate methodology-reference mode. It uses the FCED design vector `[N_TPC, N_FPC, N_TEDC, N_FEDC, N_EEDC, F_EDC, RR_EDC, N_TPDC, N_FPDC, RR_PDC, N_TSRC, N_FSRC, RR_SRC]` and minimizes `[TAC, GGE_CO2, PRI]`. It should not be confused with the operational objective. Because supplementary Table S5 was not supplied, paper-design bounds are labelled synthetic/default prototype bounds and need replacing when the supplement is available.

## How the system works

```text
Plant or simulator data
  → validation and feed-context detection
  → historical similarity / warm start
  → performance prediction or process simulation
  → NSGA-II candidate population
  → Class A/B/C evaluation
  → Pareto identification, GP-RBF identification, directed correction
  → further NSGA-II evolution
  → feasible Pareto front
  → TOPSIS balanced selection
  → hard-limit and confidence validation
  → operator recommendation
```

### Simulator and performance model

`src/process_simulator.py` defines a replaceable `ProcessSimulator`. `SyntheticFCEDSimulator` intentionally supplies qualitative behavior: increasing reflux/duties increases energy, separation improves with diminishing returns, poor stage/pressure/loading conditions reduce quality, and specified combinations cause synthetic convergence failures. It is not a physical plant model.

`AspenFCEDSimulator` is the extension point for a validated Aspen, digital twin, or hybrid model. Replacing the simulator does not change the optimizer.

`src/performance_model.py` independently defines `fit(X, y)`, `predict(X)`, and `predict_with_uncertainty(X)`. The current ensemble surrogate is only a prototype. For plant data, select and validate a model using chronological historical records.

### Constraints, NSGA-II, and Pareto front

`src/constraints.py` checks configured variable limits and purity/recovery targets. Class A violates an explicit variable, structural, purity, or recovery constraint. Class B successfully evaluates. Class C has valid variables but an evaluation failure. Only feasible Class B candidates can appear in the final Pareto front.

`src/nsga2.py` evolves a population: generate candidates, evaluate them, retain feasible good candidates, mutate, and repeat. A point is Pareto-optimal when no other point is no worse in every objective and better in at least one. Thus the output is a set of valid trade-offs, not one universal best setting.

### MO-DIDC, Gaussian Process classifier, and correction

MO-DIDC is embedded in NSGA-II; it is not “dataset → GP → one optimum.” Feasible Class B solutions are labelled Pareto (`1`) or dominated (`0`). `src/gp_classifier.py` uses an RBF-kernel `GaussianProcessClassifier` on normalized decision variables to estimate `P(Pareto | x)` for Class C points. Only candidates above the configurable threshold are promising infeasible candidates.

`src/directed_correction.py` calculates variable-wise normalized distance from feasible candidates, selects the variable with greatest average distance, samples a replacement inside the Pareto range, respects bounds/discrete values, and returns the corrected candidate to NSGA-II. `outputs/correction_matrix.csv` records every correction. It can be empty if there are insufficient GP labels or no candidate exceeds the threshold; that is intentional safe behavior.

A Gaussian Process (GP) is not a Gaussian Mixture Model (GMM). The project uses a GP classifier for Pareto-probability identification, not a GMM.

### TOPSIS

TOPSIS is a decision layer after optimization, not the optimizer. It normalizes the final feasible Pareto objective matrix, applies configurable weights, and calculates closeness to ideal and anti-ideal points. Its output is called the **TOPSIS-selected balanced Pareto operating point**, never a global optimum.

## Synthetic data and outputs

Run `scripts/generate_synthetic_data.py` to create 5,000 synthetic FCED records in `data/synthetic/fced_synthetic_5000.csv`. Records contain feed context, operating/design variables, purity, recovery, energy, throughput, operating cost, TAC, GGE_CO2, PRI, convergence status, feasibility class, and a `synthetic=True` marker.

| Output | Meaning |
| --- | --- |
| `outputs/final_pareto.csv` | Feasible nondominated operating points. |
| `outputs/balanced_solution.csv` | TOPSIS-selected balanced point. |
| `outputs/recommendation.json` | Operator-facing prototype result, marked `REQUIRES VALIDATION`. |
| `outputs/generation_history.csv` | Class and feasibility counts by generation. |
| `outputs/correction_matrix.csv` | Directed-correction audit history. |
| `outputs/gp_metrics.csv` | GP metrics or safe skip reason. |
| `outputs/optimization_summary.json` | Headline run statistics. |
| `outputs/*.png` | Pareto and convergence figures. |

## Run the synthetic workflow

Use PowerShell from the repository root:

```powershell
pip install -r requirements.txt
python scripts/generate_synthetic_data.py
python scripts/train_model.py
python scripts/train_gp.py
python scripts/run_nsga2.py
python scripts/run_modidc.py
```

Then launch the dashboard:

```powershell
python scripts/gradio_app.py
```

Open the local URL printed by Gradio, normally `http://127.0.0.1:7860`. It displays the recommendation, balanced solution, full Pareto table, GP diagnostics, correction matrix, history, plots, and synthetic data preview. Select **Reload latest optimization outputs** after a new CLI run.

## Real CSV/XLSX workflow

The real-data command validates the given file; it does not silently replace it with synthetic data or create a plant-approved recommendation.

```powershell
# Loader demonstration with included synthetic data
python scripts/run_real_data.py --file .\data\synthetic\fced_synthetic_5000.csv --mode adaptive_operational

# Real files, once placed in data/raw
python scripts/run_real_data.py --file .\data\raw\plant_data.csv --mode adaptive_operational
python scripts/run_real_data.py --file .\data\raw\plant_data.xlsx --mode adaptive_operational
```

Required feed fields are `feed_MeOH`, `feed_EtAC`, `feed_Water`, `feed_flow`, `feed_pressure`, and `feed_temperature`. The loader checks required fields, missing values, duplicates, nonnegative compositions, composition sums, and optional timestamps. For time series, use chronological train/validation/test splits and fit scalers only on the training data to prevent future-data leakage.

## Adaptation, retraining, and safety

`src/feed_similarity.py` uses normalized Euclidean distance to find comparable historical feed contexts. Similar cases can warm-start a search but do not prove the same optimum applies. `src/retraining.py` supports retraining by record count, feed shift, model-performance decline, or manual request. `src/model_registry.py` writes versioned models and metadata; production promotion needs independent validation.

Before a real recommendation is presented, validate process/equipment limits, pressure/flow/duty limits, purity/recovery requirements, current safety interlocks, model uncertainty, extrapolation outside training domain, data quality, independent plant/simulator validation, and operator approval. If any validation is uncertain, keep the status `REQUIRES VALIDATION`.

## Project structure

- `config/`: operating targets and prototype bounds.
- `data/`: raw, processed, and synthetic data.
- `models/`: versioned model artifacts and metadata.
- `outputs/`: optimization CSV/JSON files and plots.
- `scripts/`: command-line runners and the Gradio dashboard.
- `src/`: simulator, models, constraints, NSGA-II, MO-DIDC, TOPSIS, and support modules.
- `tests/`: core automated tests.

The core idea is: **data/context → predicted or simulated performance → NSGA-II + MO-DIDC → feasible Pareto front → TOPSIS selection → safety validation → operator decision support**. It is not simply a model that returns one reflux ratio.

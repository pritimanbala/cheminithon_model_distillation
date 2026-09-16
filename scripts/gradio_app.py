"""Local visual dashboard for synthetic FCED optimization artifacts."""
import sys
sys.path.insert(0, ".")
from pathlib import Path
import json
import pandas as pd
import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SYNTHETIC = ROOT / "data" / "synthetic" / "fced_synthetic_5000.csv"

def frame(name):
    path = OUT / name
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()

def obj(name):
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else {"status": "Run python scripts/run_modidc.py first."}

def image(name):
    path = OUT / name
    return str(path) if path.exists() else None

def refresh():
    """Reload files so the dashboard reflects a newly completed CLI run."""
    return (
        obj("recommendation.json"), obj("optimization_summary.json"), obj("model_validation_report.json"),
        frame("balanced_solution.csv"), frame("final_pareto.csv"), frame("generation_history.csv"),
        frame("correction_matrix.csv"), frame("gp_metrics.csv"), image("pareto_2d.png"),
        image("throughput_energy.png"), image("energy_cost.png"), image("convergence.png"),
    )

with gr.Blocks(title="FCED Optimization Dashboard") as app:
    gr.Markdown("# FCED AI-assisted optimization dashboard\n**Synthetic decision-support prototype — never use results for automatic plant control.**")
    reload_button = gr.Button("Reload latest optimization outputs", variant="primary")
    with gr.Tab("Recommendation"):
        recommendation = gr.JSON(label="TOPSIS-selected balanced Pareto operating point")
        balanced = gr.Dataframe(label="Balanced solution", interactive=False)
        validation = gr.JSON(label="Model validation / safety status")
    with gr.Tab("Pareto front"):
        pareto = gr.Dataframe(label="Feasible Pareto solutions", interactive=False, max_height=450)
        with gr.Row():
            pareto_plot = gr.Image(label="Energy vs purity", type="filepath")
            energy_plot = gr.Image(label="Throughput vs energy", type="filepath")
        cost_plot = gr.Image(label="Energy vs operating cost", type="filepath")
    with gr.Tab("Optimization diagnostics"):
        summary = gr.JSON(label="Optimization summary")
        history = gr.Dataframe(label="Generation history", interactive=False)
        correction = gr.Dataframe(label="MO-DIDC correction matrix", interactive=False)
        gp = gr.Dataframe(label="GP-RBF classifier metrics", interactive=False)
        convergence = gr.Image(label="Convergence", type="filepath")
    with gr.Tab("Synthetic data"):
        gr.Markdown("Preview only. This file is synthetic and is not a replacement for plant data.")
        data_preview = gr.Dataframe(pd.read_csv(SYNTHETIC, nrows=100) if SYNTHETIC.exists() else pd.DataFrame(), label="First 100 synthetic records", interactive=False, max_height=450)
    outputs = [recommendation, summary, validation, balanced, pareto, history, correction, gp, pareto_plot, energy_plot, cost_plot, convergence]
    reload_button.click(refresh, outputs=outputs)
    app.load(refresh, outputs=outputs)

if __name__ == "__main__":
    app.launch()

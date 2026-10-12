import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, os

mlflow.set_experiment("artifact-logging")

with mlflow.start_run(run_name="artifact-demo"):
    # 1. log a JSON report as artifact
    report = {"accuracy": 0.92, "f1": 0.89, "roc_auc": 0.95}
    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact("eval_report.json", artifact_path="reports")

    # 2. log a matplotlib plot
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = np.arange(1, 21)
    ax.plot(epochs, 1 - 0.05*epochs + np.random.normal(0, 0.01, 20), label="train")
    ax.plot(epochs, 1 - 0.04*epochs + np.random.normal(0, 0.015, 20), label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
    ax.set_title("Training curve")
    fig.savefig("training_curve.png", dpi=100, bbox_inches='tight')
    mlflow.log_artifact("training_curve.png", artifact_path="plots")
    plt.close()

    # 3. log metrics from the report
    mlflow.log_metrics(report)
    print("Artifacts logged!")
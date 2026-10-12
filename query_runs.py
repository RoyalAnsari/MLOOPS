import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

client = MlflowClient()

# get the experiment by name
exp = client.get_experiment_by_name("sklearn-classifier")
print(f"Experiment ID: {exp.experiment_id}")

# search runs with filters
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="metrics.accuracy > 0.90",
    order_by=["metrics.roc_auc DESC"],
    max_results=10
)

print(f"\nFound {len(runs)} qualifying runs:\n")

# build a summary dataframe
rows = []
for run in runs:
    rows.append({
        "run_id":      run.info.run_id[:8],
        "status":      run.info.status,
        "n_estimators":run.data.params.get("n_estimators"),
        "max_depth":   run.data.params.get("max_depth"),
        "accuracy":    round(run.data.metrics.get("accuracy", 0), 4),
        "roc_auc":     round(run.data.metrics.get("roc_auc", 0), 4),
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))

# get the best run's full run_id
best = runs[0]
print(f"\nBest run_id: {best.info.run_id}")
print(f"Best params: {best.data.params}")

# download an artifact from the best run
# client.download_artifacts(best.info.run_id, "plots", "/tmp/artifacts")

import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# yahan Task 6 wali apni run_id paste karein
BEST_RUN_ID = "3090907682e2408fbfc097de8b6faf1b"
MODEL_NAME = "BreastCancerClassifier"

# model register karein
model_uri = f"runs:/{BEST_RUN_ID}/model"
result = mlflow.register_model(model_uri, MODEL_NAME)
print(f"Registered: {MODEL_NAME} v{result.version}")

# description add karein
client.update_model_version(
    name=MODEL_NAME,
    version=result.version,
    description="Random Forest on Breast Cancer dataset.",
)

# tags
client.set_registered_model_tag(MODEL_NAME, "framework", "sklearn")
client.set_registered_model_tag(MODEL_NAME, "owner", "mlops-lab-student")

# stages ki jagah alias (MLflow 3.x ka naya tareeqa)
client.set_registered_model_alias(MODEL_NAME, "champion", result.version)
print("Alias 'champion' set ho gaya!")
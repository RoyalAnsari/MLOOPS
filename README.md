# MLflow Lab

Hands-on MLOps exercises using MLflow 3.x and scikit-learn.

## What is covered
- Experiment tracking (params, metrics, tags)
- Logging artifacts (plots, reports)
- Training and logging scikit-learn models
- Autolog and GridSearchCV
- Querying runs with MlflowClient
- Model Registry with aliases
- Loading models with pyfunc
- Automated pipeline with a promotion gate

## Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## Run
python train_model.py
mlflow ui
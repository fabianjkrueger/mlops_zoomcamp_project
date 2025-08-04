"""
This is just a tiny script demonstating how you can load and use a model from
an MLFlow server.

Make sure you have the MLFlow server running.

How to run:
```bash
pipenv shell
python scripts/load_and_use_model_from_mlflow_server.py
```
"""

# Dependencies
import mlflow
import pandas as pd
from pathlib import Path


# Paths
PATH_REPO = Path(__file__).resolve().parent.parent
PATH_DATA_PROCESSED = PATH_REPO / "data" / "processed"

# Load data
X_test = pd.read_csv(PATH_DATA_PROCESSED / "X_test.csv")
X_test.drop(columns=["release_year", "release_month"],
    inplace=True
)

# Load and use model

# set tracking URI
mlflow.set_tracking_uri("http://localhost:5001")  # for Docker

# set model name and stage
model_name = "playtime-prediction-model"
stage = "Staging"

# define model URI and print confirmation
model_uri = f"models:/{model_name}/{stage}"
print(f"\nLoading model: {model_uri}")

# load model from registry
model = mlflow.sklearn.load_model(model_uri)

# get predictions from model
predictions = model.predict(X_test)

# print predictions
print("\nPredictions:")
print(predictions)

# I didn't include evaluation here, because this is not about evaluation models
# but about explaining how to get and use a model from an MLFlow server

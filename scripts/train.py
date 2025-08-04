"""
Train the model to be used in the project.
It really is not a good model at all, but just a placeholder to have something
to work with.

How to run:
Make sure you have the data prepared and the MLflow server running.
```bash
pipenv shell
python scripts/train.py
```
"""

# Dependencies
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
PATH_DATA_PROCESSED = REPO_ROOT / "data" / "processed"


def setup_mlflow():
    """Configure MLflow tracking and experiment."""
    # set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5001")
    
    # set experiment - the artifacts will now use the HTTP API
    mlflow.set_experiment("playtime-prediction")


def load_datasets():
    """Load processed training and test datasets."""
    X_train = pd.read_csv(PATH_DATA_PROCESSED / "X_train.csv")
    y_train = pd.read_csv(PATH_DATA_PROCESSED / "y_train.csv").squeeze()  # convert to Series
    X_test = pd.read_csv(PATH_DATA_PROCESSED / "X_test.csv")
    y_test = pd.read_csv(PATH_DATA_PROCESSED / "y_test.csv").squeeze()  # convert to Series
    
    return X_train, y_train, X_test, y_test


def get_model_params():
    """Define model hyperparameters."""
    return {
        # number of trees: don't go wild here, keep it simple
        # the model doesn't really have to perform
        # I think 100 is a good balance between speed and performance
        "n_estimators": 100,
        # keep forest in check to avoid overfitting
        # maximum depth of trees
        "max_depth": 10,
        # minimum samples to split a node
        "min_samples_split": 5,
        # minimum samples in a leaf
        "min_samples_leaf": 2,
        # features to consider for best split
        "max_features": 'sqrt',
        # for reproducibility
        "random_state": 42,
        # use all available cores
        "n_jobs": -1,
    }


def train_model(X_train, y_train, model_params):
    """Train the random forest regressor."""
    rf_regressor = RandomForestRegressor(**model_params)
    rf_regressor.fit(X_train, y_train)
    return rf_regressor


def evaluate_model(model, X_test, y_test, year=2021):
    """Evaluate the trained model on test data."""
    # filter test data for specific year (since test data contains multiple years)
    year_mask = X_test["release_year"] == year
    X_test_year = X_test[year_mask]
    y_test_year = y_test[year_mask]
    
    # drop time features for prediction (same as training data)
    X_test_features = X_test_year[[
        "main_story",
        "main_story_polled",
        "main_plus_sides",
        "main_plus_sides_polled",
    ]]
    
    # make predictions
    y_pred = model.predict(X_test_features)
    
    # calculate RMSE
    mse = mean_squared_error(y_test_year, y_pred)
    rmse = mse ** 0.5
    
    print(f"RMSE for {year}: {rmse}")
    
    return rmse, y_pred


def log_to_mlflow(model_params, model, rmse_2021):
    """Log model parameters, metrics, and artifacts to MLflow."""
    # log model parameters
    mlflow.log_params(model_params)
    mlflow.log_param("model_type", "RandomForestRegressor")
    
    # log the result
    mlflow.log_metric("rmse_2021", rmse_2021)
    
    # log the model
    mlflow.sklearn.log_model(
        model,
        "random_forest_regressor"
    )


def train_and_log_model():
    """Main training pipeline with MLflow logging."""
    # setup MLflow
    setup_mlflow()
    
    # load data
    print("Loading datasets...")
    X_train, y_train, X_test, y_test = load_datasets()
    
    # get model parameters
    model_params = get_model_params()
    
    # start MLflow run
    with mlflow.start_run() as run:
        print("Training model...")
        # train model
        rf_regressor = train_model(X_train, y_train, model_params)
        
        print("Evaluating model...")
        # evaluate model
        rmse_2021, y_pred = evaluate_model(rf_regressor, X_test, y_test, year=2021)
        
        print("Logging to MLflow...")
        # log to MLflow
        log_to_mlflow(model_params, rf_regressor, rmse_2021)
        
        # run automatically ends when exiting the context manager
        print(f"MLflow run completed: {run.info.run_id}")
        
        return rf_regressor, rmse_2021


if __name__ == "__main__":
    train_and_log_model()
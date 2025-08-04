"""
Model promotion script that finds the best model from the experiment
and promotes it to the MLflow Model Registry.

How to run:
Make sure MLflow server is running.
```bash
pipenv shell
python scripts/promote_model.py
```
"""

# Dependencies
import warnings
import mlflow
from mlflow import MlflowClient
import pandas as pd
from pathlib import Path

# suppress the pkg_resources deprecation warning from MLflow
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Configuration
EXPERIMENT_NAME = "playtime-prediction"
MODEL_NAME = "playtime-prediction-model"
METRIC_NAME = "rmse_2021"
PROMOTE_TO_STAGE = "Staging"  # or "Production"


def setup_mlflow():
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri("http://localhost:5001")
    return MlflowClient()


def get_experiment_runs(client, experiment_name):
    """Get all runs from the specified experiment."""
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found!")
            return []
        
        # get all runs from the experiment, sorted by start time (newest first)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        print(f"Found {len(runs)} runs in experiment '{experiment_name}'")
        return runs
        
    except Exception as e:
        print(f"Error getting experiment runs: {e}")
        return []


def analyze_runs(runs, metric_name):
    """Analyze runs and find the best performing model."""
    if not runs:
        print("No runs found to analyze")
        return None
    
    print(f"\nAnalyzing runs based on metric: {metric_name}")
    print("-" * 60)
    
    valid_runs = []
    
    for run in runs:
        run_id = run.info.run_id
        status = run.info.status
        metric_value = run.data.metrics.get(metric_name)
        
        print(f"Run ID: {run_id[:8]}... | Status: {status} | {metric_name}: {metric_value}")
        
        # only consider successful runs with the required metric
        if status == "FINISHED" and metric_value is not None:
            valid_runs.append({
                'run_id': run_id,
                'metric_value': metric_value,
                'run_object': run
            })
    
    if not valid_runs:
        print(f"\nNo valid runs found with metric '{metric_name}'")
        return None
    
    # find best run (lowest RMSE is better)
    best_run = min(valid_runs, key=lambda x: x['metric_value'])
    
    print(f"\nBest run found:")
    print(f"  Run ID: {best_run['run_id']}")
    print(f"  {metric_name}: {best_run['metric_value']:.4f}")
    
    return best_run


def check_existing_model(client, model_name):
    """Check if model already exists in registry."""
    try:
        registered_model = client.get_registered_model(model_name)
        print(f"\nModel '{model_name}' already exists in registry")
        
        # get all versions
        versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
        
        if versions:
            print("Existing versions:")
            for version in versions:
                print(f"  Version {version.version}: Stage '{version.current_stage}'")
        
        return registered_model, versions
        
    except Exception:
        print(f"\nModel '{model_name}' does not exist in registry yet")
        return None, []


def register_best_model(client, best_run, model_name):
    """Register the best model to the MLflow Model Registry."""
    run_id = best_run['run_id']
    metric_value = best_run['metric_value']
    
    # construct model URI
    model_uri = f"runs:/{run_id}/random_forest_regressor"
    
    try:
        # check if model already exists
        existing_model, existing_versions = check_existing_model(client, model_name)
        
        # create the registered model if it doesn't exist
        if existing_model is None:
            print(f"Creating new registered model '{model_name}'...")
            try:
                client.create_registered_model(
                    name=model_name,
                    description="Playtime prediction model using Random Forest"
                )
                print(f"Registered model '{model_name}' created successfully")
            except Exception as e:
                print(f"Error creating registered model: {e}")
                return None
        
        # register new version
        print(f"Registering model version from run {run_id[:8]}...")
        
        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
            description=f"Best model with {METRIC_NAME}={metric_value:.4f}"
        )
        
        print(f"Model registered successfully!")
        print(f"   Model: {model_name}")
        print(f"   Version: {model_version.version}")
        print(f"   Run ID: {run_id}")
        print(f"   {METRIC_NAME}: {metric_value:.4f}")
        
        return model_version
        
    except Exception as e:
        print(f"Error registering model: {e}")
        return None


def promote_model(client, model_name, version, stage):
    """Promote model to specified stage."""
    try:
        print(f"\nPromoting model to '{stage}' stage...")
        
        # transition to new stage
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True  # archive other versions in the same stage
        )
        
        print(f"Model promoted successfully!")
        print(f"   Model: {model_name} version {version}")
        print(f"   Stage: {stage}")
        
        return True
        
    except Exception as e:
        print(f"Error promoting model: {e}")
        return False


def get_model_summary(client, model_name):
    """Print summary of registered model."""
    try:
        print(f"\n" + "="*60)
        print(f"MODEL REGISTRY SUMMARY: {model_name}")
        print("="*60)
        
        # get all versions
        versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
        
        if not versions:
            print("No versions found in registry")
            return
        
        for version in sorted(versions, key=lambda x: int(x.version)):
            stage = version.current_stage
            run_id = version.run_id
            
            # get run details for metrics
            try:
                run = client.get_run(run_id)
                metric_value = run.data.metrics.get(METRIC_NAME, "N/A")
                print(f"Version {version.version}: Stage='{stage}' | {METRIC_NAME}={metric_value} | Run={run_id[:8]}...")
            except:
                print(f"Version {version.version}: Stage='{stage}' | Run={run_id[:8]}...")
        
    except Exception as e:
        print(f"Error getting model summary: {e}")


def main():
    """Main function to find and promote the best model."""
    print("Starting model promotion process...")
    
    # setup MLflow
    client = setup_mlflow()
    
    # get experiment runs
    runs = get_experiment_runs(client, EXPERIMENT_NAME)
    if not runs:
        return
    
    # analyze runs and find best model
    best_run = analyze_runs(runs, METRIC_NAME)
    if not best_run:
        return
    
    # register the best model
    model_version = register_best_model(client, best_run, MODEL_NAME)
    if not model_version:
        return
    
    # promote to staging
    success = promote_model(client, MODEL_NAME, model_version.version, PROMOTE_TO_STAGE)
    
    # print final summary
    get_model_summary(client, MODEL_NAME)
    
    if success:
        print(f"\nSuccessfully promoted best model to {PROMOTE_TO_STAGE}!")
    else:
        print(f"\nModel registered but promotion to {PROMOTE_TO_STAGE} failed")


if __name__ == "__main__":
    main()
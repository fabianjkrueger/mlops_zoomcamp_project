# MLOps Zoomcamp Project

## Setup

### Environment Setup

This project uses `pipenv` to manage dependencies and virtual environments.

#### Prerequisites
- Python 3.9+ 
- pipenv (install via `pipx install pipenv` if you don't have it)

#### Create and Activate Environment

1. **Install the exact environment from the tracked files:**
   ```bash
   pipenv install
   ```
   This reads the `Pipfile.lock` to install the exact same package versions used in development.

2. **Activate the environment:**
   ```bash
   pipenv shell
   ```
   Your prompt will change to indicate you're in the pipenv environment.

#### Working with the Environment

- **Activate environment:** `pipenv shell`
- **Deactivate environment:** `exit`
- **Install new packages:** `pipenv install package_name`
- **Install dev dependencies:** `pipenv install package_name --dev`
- **Run scripts without activating:** `pipenv run python script.py`

## Data

The project uses the "How Long to Beat Video Game Playtime Dataset" from Kaggle.

This data was selected for the following reasons:
- It contains date of release, so the data can be separated by time of release
- It has an acceptable number of samples, so models should be able to learn from
it
- It provides a good target variable to predict: the number of hours played
- The theme is kind of interesting and not too common

### Download the Dataset

1. **Make sure you're in the pipenv environment:**
   ```bash
   pipenv shell
   ```

2. **Run the download script:**
   ```bash
   python scripts/download_dataset.py
   ```

This will:
- Download the dataset from Kaggle using `kagglehub` to cache
- Copy the files to `data/raw/` in your project directory
- Print the final location of the downloaded files

### Exploratory Data Analysis

The data was explored in `notebooks/exploratory_data_analysis.ipynb`.
This notebook actually runs a full data pipeline, but it's not necessary to
replicate the results of this project, because the relevant code was extracted
into scripts.

### Prepare the Data

The code for data preparation is in `scripts/data_preparation.py`.
It was extracted and refactored from the notebook to make it easier to run.

How to run:
```bash
pipenv shell
python scripts/data_preparation.py
```

## Training the Model

This course is about machine learning **operations**, *not* machine learning
engineering.
The goal here is not to build the best model, but to build the infrastructure
around it.
Here, I will just train a very simple model without any hyperparameter tuning or
fluff, because I just need **any** model.

Still, some exploratory data analysis is needed to even understand what I can
train in the first place.
This is done in `notebooks/exploratory_data_analysis.ipynb`.

## MLflow

This project uses MLflow to track experiments and model artifacts.

I hosted an MLflow server and a PostgreSQL database locally using
Docker Compose.

The server is accessible at `http://localhost:5001`.

The code for this server is in a different repository:
[mlflow_postgresql_docker](https://github.com/fabianjkrueger/mlflow_postgresql_docker).

Follow the instructions in the repository to set up the server and database in
case you would like to reproduce the results.






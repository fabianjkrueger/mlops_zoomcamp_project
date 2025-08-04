# MLOps Zoomcamp Project

## Problem Description

Since this is a course about MLOps, the goal is not to build the best model,
or solve some fancy machine learning problem,
but to build the infrastructure around the model.

My goal here was to get a model that I could deploy as a web service as well as
in batch mode.
So, I wanted some data with time of release information.
Ideally it should be updated regularly.
This would allow me to either actually schedule batch jobs frequently or to at
least simulate the process.

I found a dataset on Kaggle that seemed to be a good fit:
[How Long to Beat Video Game Playtime Dataset](https://www.kaggle.com/datasets/the-guardian/how-long-to-beat-video-game-playtime-dataset).
It contains information about the playtime of video games, including the time of
release.
It also contains information about the main story, main plus sides, and
completionist playtime.
I decided to use the main story and main plus sides playtime as features and the
completionist playtime as the target variable.
Alongside this, it contained information about when the game was released.
This would allow me to simulate the process of scheduling batch jobs regularly.
However, it also meant that I couldn't use the time of release as a feature,
so I ended up with a model that is not very good.

But as I said, in my personal opinion, getting a great model was never the goal
of this particular project.
There are many other projects, courses and competitions for getting good models.
Here, I decided to get a bad one and focus on aspects of MLOps that actually
distinguish this particular course from the others, so I could learn something
new.

So yeah, just a little more about the "problem" it is supposed to solve:
it predicts how long people are going to need for completing a video game until
they like 100% finished every little piece of content that's still left and
really isn't much left to do.
This prediciton is done based on the time people took for completing the main
story and the side quests, as well as the number of people who submitted their
playtime.
So fundamentally it doesn't really "solve" anything I guess.
But it can be deployed and generate predictions, which is cool, so I'm happy
with it.

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

#### How to run
```bash
pipenv shell
python scripts/data_preparation.py
```

## Training the Model

This course is about machine learning **operations**, *not* machine learning
engineering.
The goal here is not to build the best model, but to build the infrastructure
around it.
Here, I just trained a very simple model without any hyperparameter tuning or
fluff, because I just need **any** model.


Training the model was initially conceptualized in the notebook
`notebooks/exploratory_data_analysis.ipynb`.

The relevant code was later extracted and refactored into the script
`scripts/train.py`.

#### How to run

Make sure you have the data prepared and the MLflow server running.

```bash
pipenv shell
python scripts/train.py
```

Model is logged to MLflow.

## Model Promotion

The best model is registered in the MLflow Model Registry.
This is done in `scripts/promote_model.py`.

#### How to run

Make sure the MLflow server is running.

```bash
pipenv shell
python scripts/promote_model.py
```

The model is promoted to the `Staging` stage.

## MLflow Server

This project uses MLflow.

I hosted an MLflow server and a PostgreSQL database locally using
Docker Compose.

The server is accessible at `http://localhost:5001`.

The code for this server is in a different repository:
[mlflow_postgresql_docker](https://github.com/fabianjkrueger/mlflow_postgresql_docker).

Follow the instructions in the repository to set up the server and database in
case you would like to reproduce the results.






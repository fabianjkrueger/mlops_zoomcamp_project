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

### Data Download

The project uses the "How Long to Beat Video Game Playtime Dataset" from Kaggle.

#### Download the Dataset

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

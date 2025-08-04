"""
Prepare data for training.
This code is extracted from the notebook
`notebooks/exploratory_data_analysis.ipynb`.
Please have a look at the notebook for more details about why the code is
written the way it is.

How to run:
```bash
pipenv shell
python scripts/data_preparation.py
```
"""

# Dependencies
import pandas as pd
from pathlib import Path

# Paths
PATH_REPO = Path(__file__).resolve().parent.parent
PATH_DATA_RAW = PATH_REPO / "data" / "raw"
PATH_DATA_PROCESSED = PATH_REPO / "data" / "processed"
PATH_GAMES = PATH_DATA_RAW / "hltb_game.csv"


def load_and_clean_data():
    """Load raw data and perform initial cleaning."""
    # load the data and have a first look at it
    data_games = pd.read_csv(PATH_GAMES)
    
    # drop duplicates, keeping the first occurrence of each name
    data_games = data_games.drop_duplicates(subset=["name"], keep='first')
    
    # select columns I want to use
    data_games = data_games[[
        "id",
        "release_year",
        "release_month",
        "main_story",
        "main_story_polled",
        "main_plus_sides",
        "main_plus_sides_polled",
        "completionist",
    ]]
    
    # drop rows with missing values
    data_games.dropna(inplace=True)
    
    return data_games


def create_train_set(data_games):
    """Create training features and labels from cleaned data."""
    # features - time of release is not needed anymore after filtering
    X_train = data_games[data_games["release_year"] <= 2020][[
        "main_story",
        "main_story_polled",
        "main_plus_sides",
        "main_plus_sides_polled",
    ]]
    
    # labels
    y_train = data_games[data_games["release_year"] <= 2020]["completionist"]
    
    return X_train, y_train


def create_test_set(data_games):
    """Create test features and labels from cleaned data."""
    # features - don't drop the time of release here to simulate scheduling on new data
    X_test = data_games[data_games["release_year"] > 2020][[
        "release_year",
        "release_month",
        "main_story",
        "main_story_polled",
        "main_plus_sides",
        "main_plus_sides_polled",
    ]]
    
    # labels
    y_test = data_games[data_games["release_year"] > 2020]["completionist"]
    
    return X_test, y_test


def save_datasets(X_train, y_train, X_test, y_test):
    """Save processed datasets to intermediate directory."""
    # ensure intermediate directory exists
    PATH_DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # save training data
    X_train.to_csv(PATH_DATA_PROCESSED / "X_train.csv", index=False)
    y_train.to_csv(PATH_DATA_PROCESSED / "y_train.csv", index=False)
    
    # save test data
    X_test.to_csv(PATH_DATA_PROCESSED / "X_test.csv", index=False)
    y_test.to_csv(PATH_DATA_PROCESSED / "y_test.csv", index=False)
    
    print(f"Datasets saved to {PATH_DATA_PROCESSED}")
    print(f"Training set: {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"Test set: {len(X_test)} samples, {X_test.shape[1]} features")


def prepare_data():
    """Main function to orchestrate the data preparation pipeline."""
    print("Loading and cleaning data...")
    data_games = load_and_clean_data()
    
    print("Creating training set...")
    X_train, y_train = create_train_set(data_games)
    
    print("Creating test set...")
    X_test, y_test = create_test_set(data_games)
    
    print("Saving datasets...")
    save_datasets(X_train, y_train, X_test, y_test)
    
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    prepare_data()
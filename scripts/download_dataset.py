import kagglehub
import shutil
import os
from pathlib import Path

PATH_REPO = Path(__file__).parent.parent
PATH_DATA = PATH_REPO / "data"
PATH_DATA_RAW = PATH_DATA / "raw"

# Download to cache
cache_path = kagglehub.dataset_download("b4n4n4p0wer/how-long-to-beat-video-game-playtime-dataset")

# Copy to your desired location
os.makedirs(PATH_DATA_RAW, exist_ok=True)
shutil.copytree(cache_path, PATH_DATA_RAW, dirs_exist_ok=True)

print(f"Files copied to: {PATH_DATA_RAW}")
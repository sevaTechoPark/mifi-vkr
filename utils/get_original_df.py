import json
from pathlib import Path
import pandas as pd

def get_original_df():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    path = data_dir / "original_data.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return pd.DataFrame(data)

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

def get_processed_df():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    path = data_dir / "procesed_data.csv"
    
    df = pd.read_csv(path, encoding="utf-8")
    df = df[['label', 'next_text']].rename(columns={'next_text': 'text'})
    df['text'] = df['text'].fillna('').astype(str)
    
    return df
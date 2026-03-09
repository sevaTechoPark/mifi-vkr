import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import joblib
from pathlib import Path

MODEL_PATH = Path("models") / "baseline_tfidf_wordchar_linearsvc.joblib"

def get_baseline_model():
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    baseline_model_path = models_dir / "baseline_tfidf_wordchar_linearsvc.joblib"

    return joblib.load(baseline_model_path)

import os
import argparse
import joblib
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


def eval_model(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "balanced_accuracy": round(balanced_accuracy_score(y_test, pred), 6),
        "macro_f1": round(f1_score(y_test, pred, average="macro", zero_division=0), 6),
    }, pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vecdir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X_train = sp.load_npz(os.path.join(args.vecdir, "X_train_hybrid.npz"))
    X_test = sp.load_npz(os.path.join(args.vecdir, "X_test_hybrid.npz"))
    y_train = pd.read_csv(os.path.join(args.vecdir, "y_train.csv")).iloc[:, 0].astype(str)
    y_test = pd.read_csv(os.path.join(args.vecdir, "y_test.csv")).iloc[:, 0].astype(str)

    models = {
        "linear_svc": LinearSVC(class_weight="balanced", max_iter=5000),
        "logreg": LogisticRegression(class_weight="balanced", max_iter=2000),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics, pred = eval_model(model, X_test, y_test)
        results.append({"model": name, **metrics})
        joblib.dump(model, os.path.join(args.outdir, f"{name}.joblib"))
        pd.Series(pred).to_csv(os.path.join(args.outdir, f"{name}_pred.csv"), index=False)
        print(name, metrics)

    pd.DataFrame(results).to_csv(os.path.join(args.outdir, "results.csv"), index=False)


if __name__ == "__main__":
    main()
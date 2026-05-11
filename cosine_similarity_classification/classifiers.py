import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def normalize_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def predict_nearest(train_embs, train_labels, query_embs):
    train_embs = normalize_rows(train_embs)
    query_embs = normalize_rows(query_embs)

    sims = cosine_similarity(query_embs, train_embs)
    best_idx = np.argmax(sims, axis=1)

    pred_labels = np.asarray(train_labels)[best_idx]
    pred_scores = sims[np.arange(len(query_embs)), best_idx]

    return pred_labels, pred_scores


def build_centroids(train_embs, train_labels):
    train_embs = normalize_rows(train_embs)
    train_labels = np.asarray(train_labels)

    classes = np.unique(train_labels)
    centroids = []

    for cls in classes:
        cls_embs = train_embs[train_labels == cls]
        centroid = cls_embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        centroids.append(centroid)

    centroids = np.vstack(centroids)
    return classes, centroids


def predict_centroid(train_embs, train_labels, query_embs):
    query_embs = normalize_rows(query_embs)
    classes, centroids = build_centroids(train_embs, train_labels)

    sims = cosine_similarity(query_embs, centroids)
    best_idx = np.argmax(sims, axis=1)

    pred_labels = classes[best_idx]
    pred_scores = sims[np.arange(len(query_embs)), best_idx]

    return pred_labels, pred_scores, classes, centroids
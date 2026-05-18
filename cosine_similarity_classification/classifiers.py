import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier


def normalize_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def predict_nearest(train_embs, train_labels, query_embs, k: int = 5):
    """
    kNN на L2-нормированных эмбеддингах с косинусной метрикой и distance-weighted голосованием.

    Устойчив к дубликатам в train (когда два почти одинаковых текста имеют разные метки —
    старый argmax по single-nearest давал случайный из них; теперь голосование по k=5).
    """
    train_embs = normalize_rows(train_embs).astype(np.float32)
    query_embs = normalize_rows(query_embs).astype(np.float32)
    train_labels = np.asarray(train_labels)

    # k не должен превышать размер train
    effective_k = min(k, len(train_embs))
    if effective_k < 1:
        raise ValueError("Train set is empty")

    clf = KNeighborsClassifier(
        n_neighbors=effective_k,
        metric="cosine",
        weights="distance",
        algorithm="brute",   # на L2-нормированных эмбеддингах brute обычно быстрее всего и точен
    )
    clf.fit(train_embs, train_labels)

    pred_labels = clf.predict(query_embs)

    # «Score» оставляем для совместимости: max cos-similarity до ближайшего соседа
    sims = cosine_similarity(query_embs, train_embs)
    pred_scores = sims.max(axis=1)

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
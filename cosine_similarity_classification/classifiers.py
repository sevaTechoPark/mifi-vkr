import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


def normalize_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


# -----------------------------------------------------------------------------
# Centroid: L2-norm → mean → L2-norm + опциональный trimmed-mean
# -----------------------------------------------------------------------------

def _trimmed_class_centroid(class_embs: np.ndarray, trim_ratio: float) -> np.ndarray:
    """
    Робастный центроид класса:
    1) считаем обычное L2-нормированное среднее;
    2) убираем `trim_ratio` доли самых дальних точек по cos-similarity к этому среднему;
    3) пересчитываем среднее по оставшимся и снова L2-нормируем.

    trim_ratio:
        0.0 — обычный mean (= старое поведение)
        0.1 — отбрасываются 10% самых дальних
        0.2 — 20% (более устойчиво к мислейблам / выбросам)
    """
    if class_embs.shape[0] <= 1 or trim_ratio <= 0.0:
        centroid = class_embs.mean(axis=0)
        return centroid / (np.linalg.norm(centroid) + 1e-12)

    # 1) начальное среднее
    init_centroid = class_embs.mean(axis=0)
    init_centroid = init_centroid / (np.linalg.norm(init_centroid) + 1e-12)

    # 2) cosine от каждой точки к init_centroid (входы уже L2-нормированы)
    sims = class_embs @ init_centroid  # (N,)
    n_keep = max(1, int(np.ceil(class_embs.shape[0] * (1.0 - trim_ratio))))
    # топ-n_keep по похожести
    keep_idx = np.argpartition(-sims, n_keep - 1)[:n_keep]
    kept = class_embs[keep_idx]

    # 3) финальный L2-нормированный mean по оставшимся
    centroid = kept.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    return centroid


def build_centroids(train_embs, train_labels, trim_ratio: float = 0.0):
    """
    Возвращает (classes, centroids) — оба L2-нормированы.
    trim_ratio=0.0 — старое поведение (обычный mean).
    """
    train_embs = normalize_rows(train_embs)
    train_labels = np.asarray(train_labels)

    classes = np.unique(train_labels)
    centroids = []
    for cls in classes:
        cls_embs = train_embs[train_labels == cls]
        c = _trimmed_class_centroid(cls_embs, trim_ratio=trim_ratio)
        centroids.append(c)
    centroids = np.vstack(centroids)
    return classes, centroids


def predict_centroid(train_embs, train_labels, query_embs, trim_ratio: float = 0.0):
    """
    Предсказание по cosine-similarity к центроидам классов.

    trim_ratio: см. _trimmed_class_centroid. Дефолт 0.0 — обратная совместимость.
    """
    query_embs = normalize_rows(query_embs)
    classes, centroids = build_centroids(train_embs, train_labels, trim_ratio=trim_ratio)

    sims = cosine_similarity(query_embs, centroids)
    best_idx = np.argmax(sims, axis=1)

    pred_labels = classes[best_idx]
    pred_scores = sims[np.arange(len(query_embs)), best_idx]
    return pred_labels, pred_scores, classes, centroids


# -----------------------------------------------------------------------------
# Nearest: top-k soft-voting с температурой по cosine-similarity
# -----------------------------------------------------------------------------

def predict_nearest(train_embs, train_labels, query_embs, k: int = 5, temperature: float = 0.1):
    """
    Soft-vote по top-k соседям с температурой по cosine-similarity.

    Вместо sklearn KNeighborsClassifier(weights="distance"), который использует
    1/(distance+eps), мы суммируем softmax(cos_sim / T) по соседям каждой метки.
    Это:
      - управляемая «острота» голоса (T мала → почти argmax, T большая → почти равные голоса);
      - корректно работает с косинусной метрикой;
      - устойчиво к шумным эмбеддингам.

    k: число соседей (если >= len(train), берётся len(train)).
    temperature: чем меньше — тем больше веса самому ближнему соседу.
        0.05 — почти argmax (≈ k=1)
        0.10 — баланс между близкими и средне-близкими
        0.20 — мягкое усреднение топ-k
    """
    train_embs = normalize_rows(train_embs).astype(np.float32)
    query_embs = normalize_rows(query_embs).astype(np.float32)
    train_labels = np.asarray(train_labels)

    effective_k = min(int(k), len(train_embs))
    if effective_k < 1:
        raise ValueError("Train set is empty")

    # cosine NN на L2-нормированных эмбеддингах == argsort по dot product
    sims = query_embs @ train_embs.T  # (Q, T)
    # топ-k индексов по убыванию похожести
    topk_idx = np.argpartition(-sims, effective_k - 1, axis=1)[:, :effective_k]
    # реальные значения cos-sim для top-k
    rows = np.arange(sims.shape[0])[:, None]
    topk_sims = sims[rows, topk_idx]

    # softmax-веса по similarity / T
    weights = topk_sims / float(temperature)
    weights = weights - weights.max(axis=1, keepdims=True)  # числ. устойчивость
    weights = np.exp(weights)
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-12)

    # голосование: суммируем softmax-веса по меткам соседей
    topk_labels = train_labels[topk_idx]  # (Q, k)
    classes = np.unique(train_labels)
    label_to_col = {lab: j for j, lab in enumerate(classes)}

    Q = sims.shape[0]
    votes = np.zeros((Q, len(classes)), dtype=np.float32)
    for j in range(effective_k):
        for q in range(Q):
            votes[q, label_to_col[topk_labels[q, j]]] += weights[q, j]

    best_col = np.argmax(votes, axis=1)
    pred_labels = classes[best_col]

    # score = max весовой массы за победителя
    pred_scores = votes[np.arange(Q), best_col]
    return pred_labels, pred_scores